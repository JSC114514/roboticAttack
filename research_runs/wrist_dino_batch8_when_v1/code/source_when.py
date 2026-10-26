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
    x: Tensor of shape [B, 6, H, W]
    mean_list, std_list: lists of two tensors, each of shape [3]
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
    x: Tensor of shape [B, 6, H, W]
    mean_list, std_list: lists of two tensors, each of shape [3]
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
    def __init__(self, vla, processor, save_dir="", optimizer="pgd",resize_patch=False, dataset=""):
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
        torch.save(self.loss_buffer, '%s/loss' % (self.save_dir))

    def patchattack_unconstrained(self, train_dataloader, val_dataloader, num_iter=5000, target_action=np.zeros(7),
                                  patch_size=[3, 50, 50], lr=1 / 255, accumulate_steps=1, maskidx=[], warmup=20,
                                  filterGripTrainTo1=False, geometry=False,innerLoop=1,args=None):
        self.val_CE_loss = []
        self.val_MSE_Distance = []
        self.val_UAD = []
        self.train_CE_loss = []
        self.train_MSE_distance_loss = []
        self.train_UAD = []


        patch = torch.rand(patch_size).to(self.vla.device)
        patch.requires_grad_(True)
        patch.retain_grad()
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
        
        

        for i in tqdm(range(num_iter)):
            criterion = BiContrastiveLoss(initial_temp=0.07, final_temp=0.07, total_batches=innerLoop)
            if args.l2:
                loss_fn = nn.MSELoss(reduction='mean')
            else:
                loss_fn = nn.L1Loss(reduction='mean')
            # criterion_out = BiContrastiveLoss(initial_temp=0.07, final_temp=0.07) 
            train_relative_distance = {f"{idx}": [] for idx in maskidx}
            data = next(train_iterator)
            if len(maskidx) == 1 and maskidx[0] == 6 and filterGripTrainTo1:
                labels, attention_mask, input_ids, pixel_values = self.filter_train(data)
            else:
                pixel_values = data["pixel_values"]
                labels = data["labels"].to(self.vla.device)
                attention_mask = data["attention_mask"].to(self.vla.device)
                input_ids = data["input_ids"].to(self.vla.device)

            # print("masking labels...")
            labels = self.mask_labels(labels, maskidx)
            if i == 0:
                print("input_ids text:")
                for input_id in input_ids:
                    text = self.base_tokenizer.convert_ids_to_tokens(input_id)
                    print(text)

            with torch.no_grad():
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

            for inner_loop in range(innerLoop):
                
                if geometry:
                    modified_images, patch_mask, unnorm_images = self.randomPatchTransform.apply_random_patch_batch(pixel_values, patch, 
                                                                                         mean=self.mean,
                                                                                         std=self.std,
                                                                                         geometry=geometry, dataset=self.dataset, return_unnorm=True)
                if args.dual:
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
                loss_con = -criterion(adv_features, clean_features.detach(), bicos=1, single=True)
                loss_1 = -loss_fn(adv_features, clean_features.detach())
                p = output.projector_features.shape[1]
                vis_mask_token = self.downsample_mask_to_vis_tokens(patch_mask, p)
                loss_attn = self.attention_hijack_loss(clean_output, output, vis_mask_token, p, take_last_n=args.last_n, lambda_nonpatch=args.nonpatch, topk_txt_ratio = args.topk_text, margin= args.margin)
                loss_text = self.text_similarity_loss(output, phrases=args.phrase, vis_mask_token=vis_mask_token, tau=args.tau, beta_neg=args.beta_neg)
                loss = args.lambda_con * loss_con + args.lambda_attn * loss_attn + args.lambda_text * loss_text + args.lambda_1 * loss_1

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
                if (i + 1) % accumulate_steps == 0 or (i + 1) == len(train_dataloader):
                    scheduler.step()
            

            if i % 100 == 0:
                self.plot_loss()

            if i % 100 == 0:
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

    def downsample_mask_to_vis_tokens(self, mask, vis_tokens_len):
        """
        把像素级 mask -> 视觉 token 网格 (P=vis_tokens_len).
        假设视觉encoder是 ViT 栅格，P = gh*gw，其中 gh=gw=int(sqrt(P)).
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
        mask = labels > self.action_tokenizer.action_token_begin_idx
        masked_labels = labels[mask]
        masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
        template_labels = torch.ones_like(masked_labels,device=masked_labels.device)*-100
        for idx in maskidx:
            template_labels[:, idx] = masked_labels[:, idx]
        labels[labels > 2] = template_labels.view(-1)
        return labels

    def weighted_loss(self, logits, labels, maskid):
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
    def __init__(self, initial_temp=1, final_temp=0.07, total_batches=5):
        super(BiContrastiveLoss, self).__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = self.calculate_decay_rate()
 
    def forward(self, clean_feat, adv_feat, bicos, single=False):
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
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)
        if self.current_temp <= 0.02:
            self.current_temp = 0.02

    def calculate_decay_rate(self):
        return -np.log(self.final_temp / self.initial_temp) / self.total_batches    #0.23
