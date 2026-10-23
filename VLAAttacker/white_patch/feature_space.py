import torch
import torchvision
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import random
import math
import torch.nn.functional as F
from white_patch.appply_random_transform import RandomPatchTransform
from tqdm import tqdm
import os
import transformers
import pickle
import torch.nn as nn
from torchvision.utils import save_image
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
IGNORE_INDEX = -100

# 这个文件是 WHEN / UPA-RFAS 训练补丁的核心实现。
# 上游 scripts/when/train_sim_patch.py 已经完成两件事：
# 1. 加载并冻结 source OpenVLA 模型；
# 2. 构造 train_dataloader / val_dataloader。
# 本文件从 OpenVLAAttacker.patchattack_unconstrained 开始，只优化一个可学习的 RGB patch，
# 通过 feature deviation、repulsive contrastive、PAD、PSM 和 RAUP 内层扰动共同形成攻击目标。
#
# 完整函数调用链：
# sbatch scripts/slurm/when/train_libero10_long_patch.sbatch
#   -> scripts/when/train_sim_patch.py::main()
#      -> _train_when_patch()
#         -> VLAAttacker.white_patch.openvla_dataloader.get_dataloader()
#         -> OpenVLAAttacker(...)
#         -> OpenVLAAttacker.patchattack_unconstrained(...)
#            -> RandomPatchTransform.clean_process(...)
#            -> RandomPatchTransform.apply_random_patch_batch(...)
#            -> OpenVLA forward(clean / adv / RAUP)
#            -> BiContrastiveLoss + attention_hijack_loss + text_similarity_loss + feature loss
#            -> optimizer.step() 只更新 patch。

# def normalize(images,mean,std):
#     images = images - mean[None, :, None, None]
#     images = images / std[None, :, None, None]
    # return images

# def denormalize(images,mean,std):
#     images = images * std[None, :, None, None]
#     images = images + mean[None, :, None, None]
#     return images
def normalize(x, mean_list, std_list):
    """
    将已经处在 [0,1] RGB 空间的图像重新归一化回 OpenVLA 输入空间。

    这里假设输入是 6 通道：[前 3 个通道使用 SigLIP/视觉 backbone 的均值方差，
    后 3 个通道使用第二套 processor normalization]。RAUP 内层扰动是在 raw-like
    [0,1] 空间里加的，所以 forward 模型前必须再 normalize，否则模型看到的分布会错。
    """
    assert x.shape[1] == 6, "Expect 6-channel input"
    
    # 拆分前3通道和后3通道
    x1, x2 = x[:, :3], x[:, 3:]
    
    mean1, mean2 = mean_list[0].to(x.device).view(1, 3, 1, 1), mean_list[1].to(x.device).view(1, 3, 1, 1)
    std1, std2 = std_list[0].to(x.device).view(1, 3, 1, 1), std_list[1].to(x.device).view(1, 3, 1, 1)
    
    # 归一化
    x1 = (x1 - mean1) / std1
    x2 = (x2 - mean2) / std2
    
    # 拼回去
    x_denorm = torch.cat([x1, x2], dim=1)
    return x_denorm

def denormalize(x, mean_list, std_list):
    """
    将 OpenVLA processor 后的 6 通道输入反归一化到 [0,1] RGB 空间。

    这样做是为了让 RAUP 的小扰动 eps=2/255 有明确的像素意义，也方便保存可视化图片。
    如果直接在 normalized 空间加 eps，扰动强度会被不同通道的 std 放大或缩小。
    """
    assert x.shape[1] == 6, "Expect 6-channel input"
    
    # 拆分前3通道和后3通道
    x1, x2 = x[:, :3], x[:, 3:]
    
    mean1, mean2 = mean_list[0].to(x.device).view(1, 3, 1, 1), mean_list[1].to(x.device).view(1, 3, 1, 1)
    std1, std2 = std_list[0].to(x.device).view(1, 3, 1, 1), std_list[1].to(x.device).view(1, 3, 1, 1)
    
    # 反归一化
    x1 = x1 * std1 + mean1
    x2 = x2 * std2 + mean2
    
    # 拼回去
    x_denorm = torch.cat([x1, x2], dim=1)
    return x_denorm



class OpenVLAAttacker(object):
    """封装 WHEN 补丁训练状态。

    这个类不负责加载模型和数据集；它接收已经加载好的 VLA、processor 和 dataloader。
    这样 source OpenVLA、数据发现、Slurm 环境这些外部差异都留在入口脚本处理，
    这里专注实现论文式的 universal patch 优化。
    """

    def __init__(self, vla, processor, save_dir="", optimizer="pgd",resize_patch=False, dataset=""):
        """保存模型、tokenizer、图像变换和训练日志状态。

        vla 在入口脚本里已经 eval() 并冻结参数；这里再次 eval() 是防御式写法，
        目的是确保训练时梯度只回到 patch，而不是更新模型。
        """
        self.vla = vla.eval()
        # self.vla.vision_backbone_requires_grad = True
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.image_transform = self.processor.image_processor.apply_transform
        self.base_tokenizer = self.processor.tokenizer
        self.predict_stop_token: bool = True
        self.pad_token_id = 32000
        self.model_max_length = 2048
        self.loss_buffer = []
        self.save_dir = save_dir
        self.adv_action_L1_loss = []
        self.avg_angle_loss = []
        self.avg_distance_loss = []
        self.avg_reserve_loss = []
        self.min_val_avg_CE_loss = 1000000
        self.min_val_avg_L1_loss = 1000000
        self.max_relative_distance = -1000000
        self.reverse_direction_loss = 100000
        self.randomPatchTransform = RandomPatchTransform(self.vla.device,resize_patch)
        # WHEN 的 patch 训练需要在“可视化/像素空间”和“模型归一化空间”之间来回切换。
        # 因此这里显式保存两套 normalization 参数，供 apply_random_patch_batch、
        # normalize、denormalize 和保存图片时复用。
        self.mean = [torch.tensor([0.484375, 0.455078125, 0.40625]), torch.tensor([0.5, 0.5, 0.5])]
        self.std = [torch.tensor([0.228515625, 0.2236328125, 0.224609375]), torch.tensor([0.5, 0.5, 0.5])]
        self.optimizer = optimizer
        self.MSE_Distance_best = 10000
        self.dataset = dataset

        self.input_sizes = [[3, 224, 224], [3, 224, 224]]
        self.tvf_resize_params = [
            {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]},
            {'antialias': True, 'interpolation': 3, 'max_size': None, 'size': [224, 224]}
        ]
        self.tvf_crop_params = [
            {'output_size': [224, 224]},
            {'output_size': [224, 224]}
        ]
        self.tvf_normalize_params = [
            {'inplace': False, 'mean': [0.484375, 0.455078125, 0.40625],
             'std': [0.228515625, 0.2236328125, 0.224609375]},
            {'inplace': False, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        ]

    def plot_loss(self):
        """保存训练 loss 曲线。

        这个函数不是攻击目标本身的一部分，只是把每次 inner-loop 的 loss_buffer 写到磁盘，
        方便训练中途判断 patch 是否还在学习，或者是否出现数值发散。
        """
        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.save_dir))
        plt.clf()
        inner_loop = int(getattr(self, "inner_loop_for_plot", 1))
        if inner_loop > 1 and num_iters >= inner_loop:
            y = np.asarray(self.loss_buffer, dtype=np.float32)
            usable = (len(y) // inner_loop) * inner_loop
            outer_mean = y[:usable].reshape(-1, inner_loop).mean(axis=1)
            plt.plot(list(range(len(outer_mean))), outer_mean, label='Outer Mean Loss')
            plt.title('Outer Mean Loss Plot')
            plt.xlabel('Outer Iters')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.savefig('%s/loss_curve_outer_mean.png' % (self.save_dir))
            plt.clf()
        torch.save(self.loss_buffer, '%s/loss' % (self.save_dir))

    def patchattack_unconstrained(self, train_dataloader, val_dataloader, num_iter=5000, target_action=np.zeros(7),
                                  patch_size=[3, 50, 50], lr=1 / 255, accumulate_steps=1, maskidx=[], warmup=20,
                                  filterGripTrainTo1=False, geometry=False,innerLoop=1,args=None):
        """训练一个无约束位置的通用视觉补丁。

        这里的“无约束”指 patch 像素本身只被 clamp 到 [0,1]，不是从物理贴片材质或场景几何推导出来。
        每个 outer iteration 取一个 batch，先做 clean forward 得到参考特征，再构造带 patch 的 adv forward。
        最终优化目标由四类 WHEN 组件组成：
        - feature deviation：让 adv projector features 偏离 clean features；
        - repulsive contrastive：用反向 InfoNCE 拉开 clean/adv token 对齐关系；
        - PAD：诱导文本到视觉的注意力集中到 patch token；
        - PSM：让 patch token 靠近 probe phrase，远离当前 instruction 表征。
        如果 args.dual 打开，还会先做 RAUP 内层扰动，让 patch 在非 patch 区域扰动下仍有效。
        """
        # [WHEN-0: 状态初始化]
        # 这些列表记录训练和验证指标。它们不参与反传，只用于保存曲线和后续排查训练是否异常。
        self.val_CE_loss = []
        self.val_MSE_Distance = []
        self.val_UAD = []
        self.train_CE_loss = []
        self.train_MSE_distance_loss = []
        self.train_UAD = []

        # [WHEN-1: 创建可学习 patch]
        # patch 是唯一真正被 optimizer 更新的参数。模型参数在入口脚本中已经冻结；
        # retain_grad 方便后面读取 patch.grad 做调试和确认梯度没有断。
        patch = torch.rand(patch_size).to(self.vla.device)
        patch.requires_grad_(True)
        patch.retain_grad()
        # AdamW + cosine warmup 是 WHEN 默认训练配置。scheduler 只调 patch optimizer，
        # 所以学习率变化不会影响 frozen OpenVLA。
        if self.optimizer == "adamW":
            optimizer = transformers.AdamW([patch], lr=lr)
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=int(num_iter / accumulate_steps),
                num_cycles=0.5,
                last_epoch=-1,
            )
        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader)
        save_interval = max(1, int(getattr(args, "save_every", 100)))
        self.inner_loop_for_plot = innerLoop
        
        

        # [WHEN-2: outer iteration]
        # 每个 i 对应论文训练里的一个 outer step；在当前配置里通常是 2000 步。
        # 每个 outer step 内部会做一次 clean forward、可选 RAUP inner minimization，
        # 再做 innerLoop 次 patch 更新。
        for i in tqdm(range(num_iter)):
            # 每个 outer iteration 重新构造 contrastive loss 对象，温度固定为 0.07。
            # loss_fn 用于 feature deviation，L1 更接近论文默认；args.l2 打开时改用 MSE。
            criterion = BiContrastiveLoss(initial_temp=0.07, final_temp=0.07, total_batches=innerLoop)
            if args.l2:
                loss_fn = nn.MSELoss(reduction='mean')
            else:
                loss_fn = nn.L1Loss(reduction='mean')
            # criterion_out = BiContrastiveLoss(initial_temp=0.07, final_temp=0.07) 
            train_relative_distance = {f"{idx}": [] for idx in maskidx}
            # [WHEN-3: 取 batch]
            # dataloader 给出 OpenVLA 训练样本：pixel_values、input_ids、attention_mask、labels。
            # 这里没有重新采样数据增强；patch 的几何变化由 RandomPatchTransform 负责。
            data = next(train_iterator)
            if len(maskidx) == 1 and maskidx[0] == 6 and filterGripTrainTo1:
                labels, attention_mask, input_ids, pixel_values = self.filter_train(data)
            else:
                pixel_values = data["pixel_values"]
                labels = data["labels"].to(self.vla.device)
                attention_mask = data["attention_mask"].to(self.vla.device)
                input_ids = data["input_ids"].to(self.vla.device)

            # print("masking labels...")
            # OpenVLA 的动作被离散成 action tokens。mask_labels 只保留 maskidx 指定的动作维度，
            # 这样 loss/验证指标可以聚焦在指定动作通道，而非所有 token。
            labels = self.mask_labels(labels, maskidx)
            if i == 0:
                print("input_ids text:")
                for input_id in input_ids:
                    text = self.base_tokenizer.convert_ids_to_tokens(input_id)
                    print(text)

            # [WHEN-4: clean forward]
            # clean 分支不带 patch，只生成后续所有攻击项的参考锚点。
            with torch.no_grad():
                # clean branch 是 WHEN 的参考锚点：它提供没有 patch 时的 projector_features 和 attentions。
                # 所有攻击项都围绕 clean vs adv 的差异构造，所以 clean forward 必须先做且 detach。
                clean_im = self.randomPatchTransform.clean_process(pixel_values, mean=self.mean, std=self.std)
                clean_output: CausalLMOutputWithPast = self.vla(
                    input_ids=input_ids.to(self.vla.device),
                    attention_mask=attention_mask.to(self.vla.device),
                    pixel_values=clean_im.to(torch.bfloat16).to(self.vla.device),
                    labels=labels,
                    output_attentions=True,)
                clean_features = clean_output.projector_features.detach()
                # print("clean features shape: ", clean_features.shape)
                torch.save(clean_features, "clean_feature_openvla_libero.pt")
                save_image(denormalize(clean_im.cpu(),self.mean,self.std)[0,0:3], "clean_im_openvla_libero.png")
            if args.dual:
                # [WHEN-5: RAUP inner minimization]
                # RAUP / dual 分支：先在非 patch 区域找一个最坏扰动 perturbation。
                # perturbation 不覆盖 patch_mask 区域，目的不是替代 patch，而是让训练出来的 patch
                # 对背景/非贴片区域的小扰动更鲁棒。
                perturbation = torch.zeros_like(clean_im, device=self.vla.device, dtype=torch.bfloat16)     #.uniform_(-args.eps, args.eps)
                perturbation.requires_grad = True
                # perturbation = torch.nn.Parameter(perturbation, requires_grad=True)     #.uniform_(-args.eps, args.eps)
                out_lr = 2 * args.eps/args.step
                # opt_out = transformers.AdamW([perturbation], lr=out_lr)
                # step = 2*args.eps/args.out_lr
                modified_images, patch_mask = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, 
                                                                                            mean=self.mean,
                                                                                            std=self.std,
                                                                                            geometry=geometry, dataset=self.dataset)
                modified_images = modified_images.detach().clone()
                modified_images = denormalize(modified_images, self.mean, self.std).clamp(0,1)  # to [0,1]
                # print("error normalize:", (normalize(denormalize(modified_images, self.mean, self.std), self.mean, self.std)- modified_images).abs().max())
                for out_loop in range(args.step):
                    # 内层 minimization 使用 sign gradient，类似 PGD。
                    # 更新方向是让非 patch 扰动降低当前 feature-space 攻击目标，
                    # 外层 patch 之后要在这个更难的输入上继续优化。
                    if perturbation.grad is not None:
                        perturbation.grad.zero_()
                    outter_output: CausalLMOutputWithPast = self.vla(
                        input_ids=input_ids.to(self.vla.device),
                        attention_mask=attention_mask.to(self.vla.device),
                        pixel_values=normalize(modified_images+perturbation*(1-patch_mask), self.mean, self.std).to(torch.bfloat16).to(torch.bfloat16).to(self.vla.device),
                        labels=labels,
                        output_attentions=True,)
                    outter_features = outter_output.projector_features
                    loss_1 = loss_fn(outter_features, clean_features.detach())
                    outter_loss = criterion(outter_features, clean_features.detach(), bicos=1, single=True)
                    outter_loss = args.lambda_con * outter_loss + args.lambda_1 * loss_1
                    outter_loss.backward()
                    
                    
                    with torch.no_grad():
                        perturbation.add_(-out_lr * perturbation.grad.sign())
                        perturbation.clamp_(-args.eps, args.eps)
                    perturbation.retain_grad()
                    
                    print("outter loop:", out_loop, " outter loss:", outter_loss.item())
                    
                del outter_output, outter_features, outter_loss #, opt_out
                adv_image = torch.clamp(modified_images + perturbation, min=0.0, max=1.0)
                save_image(adv_image[0,0:3], "clean_im_perturbed_openvla_libero.png")
                del modified_images, adv_image

            # [WHEN-6: inner patch update loop]
            # 每个 outer step 内重复 innerLoop 次更新 patch；当前配置通常为 50。
            for inner_loop in range(innerLoop):
                # 外层 patch 更新循环。每次都重新把当前 patch 贴到 batch 图像上；
                # geometry=True 时 RandomPatchTransform 会引入位置/形变，使 patch 不只记住单一像素位置。
                
                if geometry:
                    modified_images, patch_mask, unnorm_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, 
                                                                                         mean=self.mean,
                                                                                         std=self.std,
                                                                                         geometry=geometry, dataset=self.dataset, return_unnorm=True)
                if args.dual:
                    # 将 RAUP 找到的非 patch 区域扰动加回 adv image。
                    # patch 区域保持由 patch 控制，背景区域用 perturbation 提供鲁棒训练压力。
                    modified_images = modified_images + perturbation.detach()*(1-patch_mask)
                output: CausalLMOutputWithPast = self.vla(
                        input_ids=input_ids.to(self.vla.device),
                        attention_mask=attention_mask.to(self.vla.device),
                        pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                        labels=labels,
                        output_attentions=True,
                        output_hidden_states=True,
                    )   #['loss', 'logits', 'past_key_values', 'projector_features']
                adv_features = output.projector_features
                # [WHEN-7: full objective]
                # WHEN 的主攻击目标是“最大化 clean/adv 表征偏离”。
                # 这里 optimizer 默认做梯度下降，所以把 contrastive 和 feature deviation 写成负号：
                # 最小化 loss 等价于最大化 clean 与 adv 的不一致。
                loss_con = -criterion(adv_features, clean_features.detach(), bicos=1, single=True)
                loss_1 = -loss_fn(adv_features, clean_features.detach())
                p = output.projector_features.shape[1]
                vis_mask_token = self.downsample_mask_to_vis_tokens(patch_mask, p)
                # PAD 使用 clean/adv attention 的差值，奖励更多文本注意力流向 patch token，
                # 同时惩罚非 patch token 吸走注意力。
                loss_attn = self.attention_hijack_loss(clean_output, output, vis_mask_token, p, take_last_n=args.last_n, lambda_nonpatch=args.nonpatch, topk_txt_ratio = args.topk_text, margin= args.margin)
                # PSM 使用 probe phrases 构造语义吸引项，并把 patch 表征推离当前指令表征，
                # 目的是让 patch 成为一个语义上“显眼但错位”的视觉 token 集合。
                loss_text = self.text_similarity_loss(output, phrases=args.phrase, vis_mask_token=vis_mask_token, tau=args.tau, beta_neg=args.beta_neg)
                loss = args.lambda_con * loss_con + args.lambda_attn * loss_attn + args.lambda_text * loss_text + args.lambda_1 * loss_1

                # [WHEN-8: backprop to patch only]
                # 反传只会更新 patch。模型参数虽然参与 forward，但 requires_grad=False；
                # optimizer.step 后立刻 clamp，是为了保证保存和评估时 patch 仍是合法 RGB。
                loss.backward()
                self.loss_buffer.append(loss.item())
                
                print("iter:", i, " inner_loop:", inner_loop, " attack loss:", loss.item())
                print("  loss_con:", loss_con.item(), " loss_attn:", loss_attn.item(), " loss_text:", loss_text.item())

                log_patch_grad = patch.grad.detach().mean().item()
                optimizer.step()
                patch.data = patch.data.clamp(0, 1)
                optimizer.zero_grad()
                self.vla.zero_grad()
                del modified_images
            del adv_features, loss, clean_features, loss_con, vis_mask_token, loss_attn, loss_text
            if args.dual:
                del perturbation


            if self.optimizer == "adamW":
                # scheduler 按 outer iteration 推进；accumulate_steps>1 时减少 scheduler 更新频率。
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    scheduler.step()
            

            if i % save_interval == 0:
                # [WHEN-9: checkpoint / visualization]
                # 按配置保存一次 loss 曲线，避免只在训练结束后才发现异常。
                self.plot_loss()

            if i % save_interval == 0:
                # 验证段不参与训练，只用于保存当前 patch、可视化 adv image，
                # 并用动作 token 的边界距离指标粗略观察 patch 是否真的影响动作预测。
                avg_CE_loss = 0
                val_num_sample = 0
                avg_MSE_Distance = 0
                avg_UAD = 0
                relative_distance = {f"{idx}": [] for idx in maskidx}
                print("evaluating...")
                with torch.no_grad():
                    for j in tqdm(range(100)):
                        save_image(patch.detach().cpu(), f"patch_{self.dataset}_step{i}_aug_our.png")
                        try:
                            data = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            data = next(val_iterator)
                        pixel_values = data["pixel_values"]
                        labels = data["labels"].to(self.vla.device)
                        attention_mask = data["attention_mask"].to(self.vla.device)
                        input_ids = data["input_ids"].to(self.vla.device)
                        val_num_sample += labels.shape[0]
                        if geometry:
                            modified_images, _ = self.randomPatchTransform.apply_random_patch_batch(pixel_values, 
                                                                                                 patch,
                                                                                                 mean=self.mean,
                                                                                                 std=self.std,
                                                                                                 geometry=geometry, dataset=self.dataset)
                        labels = self.mask_labels(labels, maskidx)
                        output: CausalLMOutputWithPast = self.vla(
                            input_ids=input_ids.to(self.vla.device),
                            attention_mask=attention_mask.to(self.vla.device),
                            pixel_values=modified_images.to(torch.bfloat16).to(self.vla.device),
                            labels=labels,
                        )   #['loss', 'logits', 'past_key_values', 'projector_features']
                        # print("attack: ", output.projector_features.shape)  #[8, 256, 4096]
                        # val_loss, val_angle_loss, val_distance_loss = self.weighted_loss(output.logits, labels, maskidx)
                        val_MSE_Distance, val_UAD = self.weighted_loss(output.logits, labels, maskidx)
                        action_logits = output.logits[:,
                                        self.vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
                        action_preds = action_logits.argmax(dim=2)
                        action_gt = labels[:, 1:].to(action_preds.device)
                        mask = action_gt > self.action_tokenizer.action_token_begin_idx
                        continuous_actions_pred = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        relative_distance = self.calculate_relative_distance(continuous_actions_pred,
                                                                             continuous_actions_gt, maskidx,
                                                                             relative_distance)
                        avg_MSE_Distance += val_MSE_Distance.item()
                        avg_UAD += val_UAD.item()
                        avg_CE_loss += output.loss.item()
                    avg_MSE_Distance /= val_num_sample
                    avg_UAD /= val_num_sample
                    avg_CE_loss/= val_num_sample
                    log_data={}
                    log_data["VAL_MSE_Distance"] = avg_MSE_Distance
                    log_data["VAL_UAD"] = val_UAD
                    for key, value in relative_distance.items():
                        property_name = f"val_rd_{key}"
                        log_data[property_name] = sum(value) / len(value)
                    if args.wandb_project != "false":
                        wandb.log(log_data,step=i)
                    if avg_MSE_Distance < self.MSE_Distance_best:
                        self.MSE_Distance_best = avg_MSE_Distance
                        temp_save_dir = os.path.join(self.save_dir, f"{str(i)}")
                        print(f"temp_save_dir: {temp_save_dir}")
                        os.makedirs(temp_save_dir, exist_ok=True)
                        torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                        val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                        os.makedirs(val_related_file_path, exist_ok=True)
                        modified_images = self.randomPatchTransform.denormalize(
                            modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                        pil_imgs = []
                        for o in range(modified_images.shape[0]):
                            pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                            pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                            pil_imgs.append(pil_img)
                        if args.wandb_project != "false":
                            wandb.log({"AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
                    temp_save_dir = os.path.join(self.save_dir, "last")
                    os.makedirs(temp_save_dir, exist_ok=True)
                    torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))
                    val_related_file_path = os.path.join(temp_save_dir, "val_related_data")
                    os.makedirs(val_related_file_path, exist_ok=True)
                    modified_images = self.randomPatchTransform.denormalize(
                        modified_images[:, 0:3, :, :].detach().cpu(), mean=self.mean[0], std=self.std[0])
                    # upload last image to wandb
                    pil_imgs = []
                    for o in range(modified_images.shape[0]):
                        pil_img = torchvision.transforms.ToPILImage()(modified_images[o, :, :, :])
                        pil_img.save(os.path.join(val_related_file_path, f"{str(o)}.png"))
                        pil_imgs.append(pil_img)
                    if args.wandb_project != "false":
                        wandb.log({"Last_Step_AdvImg": [wandb.Image(pil_img) for pil_img in pil_imgs]})
                self.val_CE_loss.append(avg_CE_loss)
                self.val_MSE_Distance.append(avg_MSE_Distance)
                self.val_UAD.append(avg_UAD)
                self.save_info(path=self.save_dir)
                # torch.cuda.empty_cache()

        self.plot_loss()
        save_image(patch.detach().cpu(), f"patch_{self.dataset}_final_aug_our.png")
        temp_save_dir = os.path.join(self.save_dir, "last")
        os.makedirs(temp_save_dir, exist_ok=True)
        torch.save(patch.detach().cpu(), os.path.join(temp_save_dir, "patch.pt"))

    def downsample_mask_to_vis_tokens(self, mask, vis_tokens_len):
        """
        把像素级 mask -> 视觉 token 网格 (P=vis_tokens_len).
        假设视觉encoder是 ViT 栅格，P = gh*gw，其中 gh=gw=int(sqrt(P)).

        这样做的原因是 PAD 和 PSM 都运行在 projector_features / attention token 空间，
        但 patch_mask 来自图像像素空间。必须把 mask 对齐到视觉 token 网格，
        否则模型不知道哪些 feature token 对应补丁区域。
        """
        P = vis_tokens_len
        g = int(P**0.5)
        # 双线性下采样到 g x g
        m_small = F.interpolate(mask, size=(g, g), mode="bilinear", align_corners=False)  # (B,1,g,g)
        return m_small.reshape(mask.size(0), -1)  # (B, P)
    
    def attention_hijack_loss(self, clean_out, out, vis_mask_token, P, take_last_n=2, lambda_nonpatch=0.5, topk_txt_ratio = 0.3,    # 仅在最关键的文本token上做Δ
    margin= 0.10,           # 排名间隔, None 则不用
    eps= 1e-6,):
        """
        把文本→视觉 的注意力质量“拉到”补丁 token 上，并抑制非补丁区域。
        vis_mask_token: (B,P) 二值/概率，标记哪些视觉token落在补丁上

        这是 PAD（Patch Attention Dominance）的实现。攻击补丁如果只是改变局部像素，
        但语言 token 和动作查询仍然关注真实物体，攻击通常不稳定；PAD 的目的就是让
        patch 在注意力分配里占优势，使后续动作预测更容易被 patch 表征牵引。
        """
        def blocks_to_share(attn_blocks):
            # 平均 heads -> 取 文本查询 × 视觉键 -> 归一化为“份额”
            shares = []
            S = attn_blocks[0].size(-1)   # 总token数 S = P + T
            for A in attn_blocks:
                # A: (B, H, S, S) with S = P + T
                A = A.float().mean(dim=1)                   # (B, S, S)
                vis_start = S - P
                A_txt_vis = A[:, vis_start:, :vis_start]         # (B, P, T)
                share = A_txt_vis / (A_txt_vis.sum(dim=-2, keepdim=True) + eps)
                shares.append(share)
            return torch.stack(shares, dim=0).mean(dim=0)   # (B, P, T)

        # 取倒数 N 层
        # 使用最后几层而不是所有层，是因为高层 attention 更接近动作/语义决策；
        # 低层 attention 更多是局部视觉处理，对最终控制输出的解释性较弱。
        attn_c = clean_out.attentions[-take_last_n:]
        attn_p = out.attentions[-take_last_n:]

        share_c = blocks_to_share(attn_c)   # (B,P,T)
        share_p = blocks_to_share(attn_p)   # (B,P,T)
        Delta   = share_p - share_c         # (B,P,T) 份额“增量”

        B, _, T = Delta.shape
        vis_mask_bt = vis_mask_token.unsqueeze(-1).expand(-1, -1, T)  # (B,P,T)
        if topk_txt_ratio is not None and 0 < topk_txt_ratio < 1:
            # 以 clean 的文本→视觉总量来度量“关键性”
            w_txt = share_c.sum(dim=-2)                      # (B,T)
            kT = max(1, int(T * topk_txt_ratio))
            idx = w_txt.topk(kT, dim=1).indices             # (B,kT)
            txt_mask = torch.zeros_like(w_txt).scatter(1, idx, 1.0).bool()
            txt_mask = txt_mask.unsqueeze(-2).expand(-1, P, -1)
            Delta = torch.where(txt_mask, Delta, torch.zeros_like(Delta))
        # patch / nonpatch 的增量
        d_patch    = (Delta * vis_mask_bt).sum(dim=-1)                 # (B,T)
        d_nonpatch = (Delta * (1 - vis_mask_bt)).sum(dim=-1)           # (B,T)

        # 目标：补丁增量越大越好，非补丁“正增量”越小越好（负增量不惩罚）
        loss_core = -(d_patch.mean()) + lambda_nonpatch * F.relu(d_nonpatch).mean()
        # 排名间隔（可选）：要求 patch 增量 至少 比 “最强非补丁增量” 大 margin
        if margin is not None and margin > 0:
            non_top = (Delta * (1 - vis_mask_bt)).amax(dim=-1)         # (B,T)
            rank_term = F.relu(margin - (d_patch - non_top)).mean()
            loss = loss_core + rank_term
        else:
            loss = loss_core
        return loss
    
    def build_phrase_prototypes(self, phrases=("put", "pick up", "place", "turn on", "push", "open", "close"), use_static=True):
        """
        返回 (K, D) 的原型（归一化向量）。
        use_static=True: 用 input_embeddings 的静态词向量均值（快，单次）。
        use_static=False: 将短语过一遍模型（不带图像），取最后一层文本隐状态均值（更语境化）。

        这些 phrase prototypes 是 PSM 的“语义诱饵”。补丁区域视觉 token 会被拉向这些
        高频动作/空间短语，使模型把 patch 当成强语义对象，而不是普通背景噪声。
        """
        # D = model.get_input_embeddings().weight.shape[-1]
        if use_static:
            E = self.vla.get_input_embeddings().weight.detach().to(self.vla.device).float()  # (V,D)
            # print("E shape:", E.shape)    #[32064, 4096]
            prots = []
            for p in phrases:
                ids = self.base_tokenizer(p, add_special_tokens=False)["input_ids"]
                vec = E[torch.tensor(ids, device=self.vla.device)].mean(dim=0)
                prots.append(vec)
            Pk = torch.stack(prots, dim=0)  # (K,D)
            return F.normalize(Pk, dim=-1)
        else:
            # 语境化：让模型只处理文本，取 hidden_states 的文本段均值
            with torch.no_grad():
                toks = self.base_tokenizer(phrases, padding=True, return_tensors="pt").to(self.vla.device)
                out = self.vla(language_inputs=toks["input_ids"],
                            attention_mask=toks["attention_mask"],
                            return_dict=True,
                            output_hidden_states=True)
                hs_last = out.hidden_states[-1]     # (B, S, D)
                # 均值仅在非 pad 位置
                mask = toks["attention_mask"].unsqueeze(-1).float()
                vec = (hs_last * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # (K,D)
                return F.normalize(vec, dim=-1)
            
    def text_similarity_loss(self, out, phrases, vis_mask_token, tau=0.3,
                         alpha_pos=1.0, beta_neg=0.5, use_static=True):
        """
        让补丁区域视觉token的表达：
        - 正向靠近一组 probe phrases（LogSumExp 聚合）
        - 反向远离当前输入文本的整体表征

        这是 PSM（Patch Semantic Misalignment）的实现。正项让 patch 具有可吸引模型的
        通用动作语义，负项让 patch 不贴合当前真实 instruction，从而制造语义错配。
        """
        B, P, D = out.projector_features.shape
        # 补丁视觉 token 的均值表征
        vis = out.projector_features.float()                 # (B,P,D)
        m = (vis_mask_token / (vis_mask_token.sum(dim=1, keepdim=True) + 1e-6)).unsqueeze(-1)  # (B,P,1)
        v_patch = (vis * m).sum(dim=1)                       # (B,D)
        v_patch = F.normalize(v_patch, dim=-1)               # (B,D)

        # probe 短语原型 (K,D)
        Pk = self.build_phrase_prototypes(phrases, use_static=use_static)  # (K,D)
        # 正项：最大化与任一 probe 的相似度 -> 等价于最小化 -logsumexp
        pos_sims = torch.matmul(v_patch, Pk.t()) / tau       # (B,K)
        L_pos = -torch.logsumexp(pos_sims, dim=-1).mean()

        # 当前指令文本的整体表征（最后一层文本 hidden 的均值）
        hs_last = out.hidden_states[-1].float()              # (B,S,D)
        P = out.projector_features.shape[1]
        S = hs_last.shape[1]
        txt = hs_last[:, :S-P, :]                            # (B, T, D)
        v_text = txt.mean(dim=1)                             # (B,D)
        v_text = F.normalize(v_text, dim=-1)

        # 负项：远离当前文本
        L_neg = (v_patch * v_text).sum(dim=-1).mean()

        return alpha_pos * L_pos + beta_neg * L_neg

    def modifiy_labels(self, labels,
                       target_action={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}):
        """把动作 token 标签替换成指定 target_action。

        这个函数是旧的 targeted attack 辅助逻辑，当前 WHEN 训练主路径没有直接调用。
        保留它是为了兼容原仓库中基于指定动作 token 的攻击实验。
        """
        newlabels = []
        for j in range(labels.shape[0]):
            temp_label = labels[j]
            first_valid_index = (temp_label != -100).nonzero(as_tuple=True)[0].item()
            for key, value in target_action.items():
                if value != -100:
                    temp_label[int(first_valid_index + int(key))] = value
            newlabels.append(temp_label.unsqueeze(0))
            print(temp_label)
        newlabels = torch.cat(newlabels, dim=0)
        return newlabels

    def filter_train(self, data):
        """筛选 gripper 动作为指定 token 的训练样本。

        当只攻击 gripper 维度时，训练 batch 里如果没有足够的目标状态样本，
        梯度会很稀疏；这个过滤函数试图保留 gripper token 等于 31744 的样本，
        让 targeted gripper attack 更稳定。当前完整 WHEN 配置通常不启用这个分支。
        """
        pixel_values = data["pixel_values"]
        labels = data["labels"].to(self.vla.device)
        attention_mask = data["attention_mask"].to(self.vla.device)
        input_ids = data["input_ids"].to(self.vla.device)

        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        one_index = []
        for idx in range(masked_labels.shape[0]):
            if masked_labels[idx, 6] == 31744:
                one_index.append(idx)
        if 1 < len(one_index) < 8:
            labels = labels[one_index, :]
            attention_mask = attention_mask[one_index, :]
            input_ids = input_ids[one_index, :]
            pixel_values = [pixel_values[i] for i in one_index]
        elif len(one_index) > 8:
            chosen = random.sample(one_index, k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        elif one_index is None:
            chosen = random.sample(range(labels.shape[0]), k=8)
            labels = labels[chosen, :]
            attention_mask = attention_mask[chosen, :]
            input_ids = input_ids[chosen, :]
            pixel_values = [pixel_values[i] for i in chosen]
        return labels, attention_mask, input_ids, pixel_values

    def save_info(self, path):
        """把训练/验证曲线保存为 pkl。

        原 UPA-RFAS 代码用这些文件配合 loss_curve.png 观察训练过程。
        它们不影响 patch 更新，只是为了断点后分析训练状态。
        """
        with open(os.path.join(path, 'train_CE_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_CE_loss, file)
        with open(os.path.join(path, 'train_MSE_distance_loss.pkl'), 'wb') as file:
            pickle.dump(self.train_MSE_distance_loss, file)
        with open(os.path.join(path, 'train_UAD.pkl'), 'wb') as file:
            pickle.dump(self.train_UAD, file)
        with open(os.path.join(path, 'val_CE_loss.pkl'), 'wb') as file:
            pickle.dump(self.val_CE_loss, file)
        with open(os.path.join(path, 'val_MSE_Distance.pkl'), 'wb') as file:
            pickle.dump(self.val_MSE_Distance, file)
        with open(os.path.join(path, 'val_UAD.pkl'), 'wb') as file:
            pickle.dump(self.val_UAD, file)

    def calculate_relative_distance(self,pred,gt,maskidx,relative_distance):
        """计算预测动作相对真实动作边界的偏移比例。

        LIBERO/OpenVLA 动作通常归一化在 [-1,1] 附近。直接看绝对误差不能反映
        某个维度离可行边界有多远，因此这里把误差除以 gt 到更远边界的距离，
        得到一个便于跨动作维度比较的相对距离。
        """
        pred = pred.clone().view(pred.shape[0]//len(maskidx),len(maskidx))
        gt = gt.clone().view(gt.shape[0]//len(maskidx),len(maskidx))
        for idx1 in range(pred.shape[0]):
            for idx2 in range(pred.shape[1]):
                anchor = gt[idx1,idx2]
                input_point = pred[idx1,idx2]
                upper_bound = 1
                lower_bound = -1
                distance_to_upper = upper_bound - anchor
                distance_to_lower = anchor - lower_bound
                max_boundary_distance = max(distance_to_upper, distance_to_lower)
                distance_to_anchor = abs(input_point - anchor)
                temp_relative_distance = distance_to_anchor / max_boundary_distance
                relative_distance[f"{str(maskidx[idx2])}"].append(temp_relative_distance.item())
        return relative_distance

    def mask_labels(self,labels,maskidx):
        """只保留 maskidx 指定动作维度的监督标签。

        OpenVLA 的 action chunk 被编码成连续 action tokens。若所有维度都参与 loss，
        某些实验会被无关动作维度的梯度稀释。这里把未选中的动作 token 改为 -100，
        让 HuggingFace CE loss 忽略它们。
        """
        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        template_labels = torch.ones_like(masked_labels,device=masked_labels.device)*-100
        for idx in maskidx:
            template_labels[:, idx] = masked_labels[:, idx]
        labels[labels > 2] = template_labels.view(-1)
        return labels

    def weighted_loss(self, logits, labels, maskid):
        """验证时把离散 action token 转成近似连续值，计算边界距离和 UAD。

        训练主 objective 用 feature/PAD/PSM，不直接最小化这个 weighted_loss。
        这里主要用于验证：如果 patch 真的扰乱动作，动作 token 分布会向边界或错误值移动。
        """
        temp_label = labels[:,1:].to(labels.device) # (bs,seq_len) remove bos token
        action_mask = temp_label > 2
        temp_logits = logits[:,:,31744:32000] # (bs,seq_len,256) only consider the 256 classes for the target class
        action_logits = temp_logits[:,-temp_label.shape[-1]-1:-1, :] # shift logits see modeling_llama.py line 1233
        action_logits = action_logits[action_mask]
        reweigh = torch.arange(1,257).to(logits.device)/256 # [1,,...256]
        temp_prob = F.softmax(action_logits,dim=-1) # [bs,action_length,256]
        reweighted_prob = (temp_prob * reweigh).sum(dim=-1)  # [bs, action_length]
        hard_max_labels = temp_label[action_mask]
        hard_max_labels[hard_max_labels > 31872]=31999
        hard_max_labels[hard_max_labels <= 31872]=31744
        hard_max_labels[hard_max_labels == 31999]=1/256
        hard_max_labels[hard_max_labels == 31744]=1
        UAD = self.cal_UAD(action_logits.argmax(dim=-1)+31744,temp_label[action_mask])
        distance_loss = F.mse_loss(5*reweighted_prob.contiguous(), 5*hard_max_labels.float().contiguous())

        # targeted max distance CE loss
        # ce_action_logits = logits[:,-temp_label.shape[-1]-1:-1, :]
        # ce_action_logits = ce_action_logits[action_mask]
        # ce_hard_max_labels = temp_label[action_mask]
        # ce_hard_max_labels[ce_hard_max_labels > 31872]=31744
        # ce_hard_max_labels[ce_hard_max_labels <= 31872]=31999
        # ce_loss = F.cross_entropy(ce_action_logits, ce_hard_max_labels)
        # distance_loss = distance_loss + ce_loss
        return distance_loss, UAD

    def cal_UAD(self,pred,gt):
        """计算 Unnormalized Action Distance 风格的归一化动作偏移。

        它把 token 解码回连续动作，再根据真实动作离 [-1,1] 边界的距离做归一化，
        用来衡量攻击后动作预测偏离真实动作的程度。
        """
        continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(gt.clone().detach().cpu().numpy())
        )
        continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(pred.clone().detach().cpu().numpy())
        )
        max_distance = torch.where(continuous_actions_gt > 0, torch.abs(continuous_actions_gt - (-1)), torch.abs(continuous_actions_gt - 1))
        distance = torch.abs(continuous_actions_pred - continuous_actions_gt)
        UAD = (distance / max_distance).mean()
        return UAD

class BiContrastiveLoss(nn.Module):
    """WHEN 里的 repulsive contrastive loss 基础组件。

    forward 本身计算的是 clean token 与 adv token 的 InfoNCE 对齐损失。
    主训练里会对它取负号，所以优化 patch 时等价于破坏 clean/adv token 的一一对应关系。
    """

    def __init__(self, initial_temp=1, final_temp=0.07, total_batches=5):
        """保存温度调度参数。

        当前调用里 initial_temp 和 final_temp 都是 0.07，因此温度实际固定；
        这个接口保留了原仓库可退火温度的能力。
        """
        super(BiContrastiveLoss, self).__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = self.calculate_decay_rate()
 
    def forward(self, clean_feat, adv_feat, bicos, single=False):
        """计算 token-level InfoNCE。

        clean_feat 和 adv_feat 先 L2 normalize，再做批内 token 相似度矩阵。
        标签是对角线 token 匹配；训练外层取负号后，会鼓励 adv token 远离 clean token。
        """
        B, D, T = clean_feat.shape
        # print(f"feature shape: {clean_feat.shape}")
        if bicos == 1:
            clean_feat = clean_feat.reshape(B, D, T)
            adv_feat = adv_feat.reshape(B, D, T)
            clean_feat = F.normalize(clean_feat, dim=-1)
            adv_feat = F.normalize(adv_feat, dim=-1)
    
            cos_sim = torch.bmm(clean_feat, adv_feat.transpose(1,2)) / self.current_temp
            labels = torch.arange(cos_sim.size(1), device=cos_sim.device).unsqueeze(0).expand(cos_sim.size(0), -1)

            loss = F.cross_entropy(
            cos_sim.reshape(-1, cos_sim.size(-1)),  
            labels.reshape(-1),
            reduction='none'
        ).reshape(cos_sim.size(0), cos_sim.size(1)).mean(dim=1)
    
        return torch.mean(loss)

    def update_temperature(self, batch_count):
        """按指数衰减更新温度。

        温度越低，contrastive loss 越关注最相似的负样本；这里设置了 0.02 下限，
        避免温度过低导致 softmax 数值过尖。
        """
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)
        if self.current_temp <= 0.02:
            self.current_temp = 0.02

    def calculate_decay_rate(self):
        """根据初始温度、最终温度和总 batch 数计算指数衰减率。"""
        return -np.log(self.final_temp / self.initial_temp) / self.total_batches    #0.23
