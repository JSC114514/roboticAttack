import random
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

class RandomPatchTransform:
    def __init__(self, device, resize_patch):
        self.device = device
        self.angle = 30
        self.shx = 0.2
        self.shy = 0.2
        self.resize_patch = resize_patch
        self._warned_messages = set()
        self._debug_printed = set()

    def normalize(self, images, mean, std):
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self,images, mean, std):
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images

    def rotation_matrix(self,theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def shear_matrix(self,shx, shy):
        return np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
    def simulation_random_patch(self, image, patch, geometry=False,colorjitter=False,angle=1,shx=0.1,shy=0.1,position=(0,0)):
        """
        random paste patch to images

        param:
        image (numpy.ndarray):  ndarray image [224,224,3]
        patch (torch.Tensor):  [3, patch_height, patch_width] patch

        return:
        torch.Tensor: batch img with patch added
        """
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        img_channels, img_height, img_width = image.shape

        canvas = torch.ones(img_channels, img_height, img_width).to(self.device) * -100
        patch_channels, patch_height, patch_width = patch.shape
        patch = torch.from_numpy(np.asarray(torchvision.transforms.ToPILImage()(patch))).permute(2, 0, 1)

        # x = 160
        # y = 80

        x,y = position[0],position[1]
        canvas[:, y:y + patch_height, x:x + patch_width] = patch

        if geometry:
            R = self.rotation_matrix(angle)
            # T = translation_matrix(tx, ty)
            S = self.shear_matrix(shx, shy)
            # combined_matrix = np.dot(T, np.dot(S, R))
            combined_matrix = np.dot(S, R)
            affline_matrix = torch.tensor(combined_matrix)
            canvas = self.apply_affine_transform(canvas, affline_matrix)

        image = torch.where(canvas < 0, image, canvas)
        return image.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    
    def combined_transform_matrix(self):
        if np.random.rand() < 0.2:
            return torch.tensor(np.eye(3, dtype=np.float32))
        else:
            angle = np.random.uniform(-self.angle, self.angle)
            shx = np.random.uniform(-self.shx, self.shx)
            shy = np.random.uniform(-self.shy, self.shy)

            R = self.rotation_matrix(angle)
            S = self.shear_matrix(shx, shy)
            combined_matrix = np.dot(S, R)
            return torch.tensor(combined_matrix)

    def apply_affine_transform(self,image, transform_matrix):
        if image.ndim == 4:
            image = image.squeeze(0)
        affine_matrix = transform_matrix[:2, :].unsqueeze(0)  # [1, 2, 3]

        grid = F.affine_grid(affine_matrix, image.unsqueeze(0).size(), align_corners=False)

        transformed_image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False,padding_mode='border')

        return transformed_image

    def _warn_once(self, message):
        """多视角回退信息只打印一次，避免训练日志被重复 warning 淹没。"""
        if message not in self._warned_messages:
            print(message)
            self._warned_messages.add(message)

    def _parse_views(self, value):
        """解析 front,wrist 或列表形式的视角名。"""
        if value is None:
            return []
        if isinstance(value, str):
            return [view.strip() for view in value.split(",") if view.strip() and view.strip() != "none"]
        return [str(view).strip() for view in value if str(view).strip() and str(view).strip() != "none"]

    def _to_rgb_pil(self, image):
        """把 dataloader 中的 PIL/numpy 图像统一成 RGB PIL。"""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return torchvision.transforms.ToPILImage()(image).convert("RGB")

    def _paste_patch_tensor(self, image, patch, geometry):
        """在单张视角图上贴 patch，返回未归一化的 tensor 图像。"""
        im = torchvision.transforms.ToTensor()(self._to_rgb_pil(image)).to(self.device)
        img_channels, img_height, img_width = im.shape
        patch_to_apply = patch
        patch_channels, patch_height, patch_width = patch_to_apply.shape

        if self.resize_patch:
            scale = random.uniform(0.61, 1.39)  # 约对应 1%~5% patch 面积范围
            resized_height = max(1, int(patch_height * scale))
            resized_width = max(1, int(patch_width * scale))
            patch_to_apply = transforms.Resize((resized_height, resized_width))(patch_to_apply)
            patch_channels, patch_height, patch_width = patch_to_apply.shape

        if patch_height > img_height or patch_width > img_width:
            # patch 比图像大时缩到可贴范围，避免 randint 负区间导致实验直接崩溃。
            resized_height = min(patch_height, img_height)
            resized_width = min(patch_width, img_width)
            patch_to_apply = transforms.Resize((resized_height, resized_width))(patch_to_apply)
            patch_channels, patch_height, patch_width = patch_to_apply.shape

        canvas = torch.ones(img_channels, img_height, img_width).to(self.device) * -100
        max_x = img_width - patch_width
        max_y = img_height - patch_height
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        canvas[:, y:y + patch_height, x:x + patch_width] = patch_to_apply

        if geometry:
            affline_matrix = self.combined_transform_matrix().to(self.device)
            canvas = self.apply_affine_transform(canvas, affline_matrix)
            # affine_grid 返回带 batch 维的 4D tensor；这里压回单图 3D，才能和干净视角拼接。
            if canvas.ndim == 4:
                canvas = canvas.squeeze(0)

        return torch.where(canvas < -20, im, canvas).clamp(0, 1)

    def _normalize_tensor_batch(self, image_tensors, mean, std):
        """把未归一化 RGB tensor 列表转换成 OpenVLA 需要的 6 通道输入。"""
        modified_images = []
        for im in image_tensors:
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0, im1], dim=1))
        return torch.cat(modified_images, dim=0)

    def _fuse_view_tensors(self, view_tensors, fusion):
        """多视角最小融合: 水平拼接后缩回原尺寸，使 OpenVLA 仍接收单张图。"""
        if not view_tensors:
            raise ValueError("view_tensors 为空，无法构造多视角输入")
        if len(view_tensors) == 1:
            return view_tensors[0]
        if fusion != "stitch":
            self._warn_once(f"[multiview][warn] 未支持的 fusion={fusion}; 当前回退到第一个视角。")
            return view_tensors[0]

        base_height, base_width = view_tensors[0].shape[-2:]
        resized_views = []
        for tensor in view_tensors:
            if tensor.shape[-2:] != (base_height, base_width):
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(base_height, base_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            resized_views.append(tensor)
        stitched = torch.cat(resized_views, dim=2)
        return F.interpolate(
            stitched.unsqueeze(0),
            size=(base_height, base_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def apply_patch_to_selected_views(
            self,
            batch,
            patch,
            mean,
            std,
            input_views=None,
            attack_views=None,
            view_key_map=None,
            geometry=False,
            fusion="stitch",
            dry_run=False,
            phase="train",
            debug=False):
        """按 attack_views 选择视角贴 patch；识别不了多视角时回退到原单视角流程。"""
        input_view_list = self._parse_views(input_views) or self._parse_views(batch.get("input_views", [])) or ["front"]
        attack_view_set = set(self._parse_views(attack_views))
        pixel_values_by_view = batch.get("pixel_values_by_view")

        if dry_run:
            print(
                f"[multiview][dry-run] phase={phase} input_views={input_view_list} "
                f"attack_views={sorted(attack_view_set)} view_key_map={view_key_map if view_key_map else '默认映射'}"
            )
            return None

        if not isinstance(pixel_values_by_view, dict) or not pixel_values_by_view:
            if any(view not in {"front", "primary"} for view in input_view_list + list(attack_view_set)):
                self._warn_once(
                    "[multiview][warn] batch 中没有 pixel_values_by_view，无法区分 front/wrist/side；"
                    "已回退到原单视角贴 patch。"
                )
            if attack_view_set:
                return self.apply_random_patch_batch(batch["pixel_values"], patch, mean=mean, std=std, geometry=geometry)
            clean_tensors = [torchvision.transforms.ToTensor()(self._to_rgb_pil(im)).to(self.device) for im in batch["pixel_values"]]
            return self._normalize_tensor_batch(clean_tensors, mean, std)

        available_views = sorted(pixel_values_by_view.keys())
        if debug and phase not in self._debug_printed:
            print(
                f"[multiview_debug] phase={phase} input_views={input_view_list} "
                f"attack_views={sorted(attack_view_set)} available_views={available_views} fusion={fusion}"
            )
            self._debug_printed.add(phase)

        first_view = pixel_values_by_view[available_views[0]]
        batch_size = len(first_view)
        fused_images = []
        missing_attack_views = attack_view_set - set(available_views)
        if missing_attack_views:
            self._warn_once(
                f"[multiview][warn] attack_views={sorted(missing_attack_views)} 不在 batch 可用视角 {available_views} 中；"
                "这些视角本次不会被贴 patch。"
            )

        for sample_idx in range(batch_size):
            per_view_tensors = []
            for view in input_view_list:
                if view not in pixel_values_by_view:
                    self._warn_once(
                        f"[multiview][warn] input view={view} 不在 batch 可用视角 {available_views} 中；"
                        f"样本将使用 {available_views[0]} 作为回退。"
                    )
                    image = pixel_values_by_view[available_views[0]][sample_idx]
                else:
                    image = pixel_values_by_view[view][sample_idx]

                if view in attack_view_set:
                    per_view_tensors.append(self._paste_patch_tensor(image, patch, geometry=geometry))
                else:
                    per_view_tensors.append(torchvision.transforms.ToTensor()(self._to_rgb_pil(image)).to(self.device))

            fused_images.append(self._fuse_view_tensors(per_view_tensors, fusion=fusion))
        return self._normalize_tensor_batch(fused_images, mean, std)

    def apply_random_patch_batch(self, images, patch, mean, std,geometry,colorjitter=False):
        # colorjitter 当前仅为兼容已有调用预留, 暂不改变贴图行为。
        patched_images = [self._paste_patch_tensor(im, patch, geometry=geometry) for im in images]
        return self._normalize_tensor_batch(patched_images, mean, std)

    def random_paste_patch(self, images, patch, mean, std):
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            max_x = img_width - patch_width
            max_y = img_height - patch_height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            im = torch.where(canvas != -100, canvas, im)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)

    def paste_patch_fix(self, images, patch, mean, std, inference=False):

        canvas_list = []
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            img_channels, img_height, img_width = im.shape

            canvas = torch.ones(img_channels, img_height, img_width).to(self.device)*-100
            patch_channels, patch_height, patch_width = patch.shape

            max_x = img_width - patch_width
            max_y = img_height - patch_height

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            canvas[:, y:y + patch_height, x:x + patch_width] = patch

            im = torch.where(canvas != -100, canvas, im)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))

            modified_images.append(torch.cat([im0,im1],dim=1))
            canvas_list.append(canvas)
        if inference:
            return torch.cat(modified_images, dim=0), canvas_list
        else:
            return torch.cat(modified_images, dim=0)

    def im_process(self, images, mean, std):
        modified_images = []
        for im in images:
            im = torchvision.transforms.ToTensor()(im).to(self.device)
            im0 = self.normalize(im, mean[0].to(self.device), std[0].to(self.device))
            im1 = self.normalize(im, mean[1].to(self.device), std[1].to(self.device))
            modified_images.append(torch.cat([im0,im1],dim=1))
        return torch.cat(modified_images, dim=0)
