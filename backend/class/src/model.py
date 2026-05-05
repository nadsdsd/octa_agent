from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .lora import apply_lora_by_suffix
from .mamba_fusion import MambaFusionHead

class Qwen2VLImageClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        dropout: float = 0.1,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        freeze_visual: bool = True,
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: Tuple[str, ...] = (),
        num_metrics: int = 0,
        metrics_hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_metrics = int(num_metrics)
        self.metrics_hidden_dim = int(metrics_hidden_dim)

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=False,
            trust_remote_code=False,
        )

        self.base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )

        if hasattr(self.base, "visual"):
            self.visual = self.base.visual
        elif hasattr(self.base, "model") and hasattr(self.base.model, "visual"):
            self.visual = self.base.model.visual
        else:
            raise AttributeError("Cannot find visual tower on Qwen2VLForConditionalGeneration.")

        if hasattr(self.visual, "config") and hasattr(self.visual.config, "spatial_merge_size"):
            self.spatial_merge_size = int(self.visual.config.spatial_merge_size)
        elif hasattr(self.base.config, "vision_config") and hasattr(self.base.config.vision_config, "spatial_merge_size"):
            self.spatial_merge_size = int(self.base.config.vision_config.spatial_merge_size)
        else:
            self.spatial_merge_size = 2

        if hasattr(self.visual, "config") and hasattr(self.visual.config, "hidden_size"):
            hidden_size = int(self.visual.config.hidden_size)
        elif hasattr(self.base.config, "vision_config") and hasattr(self.base.config.vision_config, "hidden_size"):
            hidden_size = int(self.base.config.vision_config.hidden_size)
        else:
            raise AttributeError("Cannot determine visual hidden_size from model config.")

        if freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad = False

        if self.num_metrics > 0:
            self.metrics_mlp = nn.Sequential(
                nn.Linear(self.num_metrics, self.metrics_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            final_dim = hidden_size + self.metrics_hidden_dim
        else:
            self.metrics_mlp = None
            final_dim = hidden_size

        self.norm = nn.LayerNorm(final_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_dim, num_classes)

        self.replaced_lora_modules: List[str] = []
        if use_lora:
            self.replaced_lora_modules = apply_lora_by_suffix(
                self.visual,
                target_suffixes=lora_targets,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.visual_is_trainable = any(p.requires_grad for p in self.visual.parameters())

    def encode_images(self, images: List[Image.Image]):
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")

        device = next(self.parameters()).device
        pixel_values = image_inputs["pixel_values"].to(device)
        image_grid_thw = image_inputs["image_grid_thw"].to(device)

        try:
            visual_dtype = next(self.visual.parameters()).dtype
        except StopIteration:
            visual_dtype = pixel_values.dtype

        pixel_values = pixel_values.to(visual_dtype)

        grad_context = nullcontext() if self.visual_is_trainable else torch.no_grad()
        with grad_context:
            outputs = self.visual(pixel_values, grid_thw=image_grid_thw)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_tokens = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            image_tokens = outputs.last_hidden_state
        elif torch.is_tensor(outputs):
            image_tokens = outputs
        elif isinstance(outputs, tuple):
            image_tokens = outputs[0]
        else:
            raise RuntimeError("Cannot parse visual outputs from Qwen2-VL visual encoder.")

        return image_tokens, image_grid_thw

    def _split_tokens_by_image(self, image_tokens: torch.Tensor, image_grid_thw: torch.Tensor):
        counts = []
        for t, h, w in image_grid_thw.tolist():
            token_count = int(t * (h // self.spatial_merge_size) * (w // self.spatial_merge_size))
            counts.append(token_count)

        chunks = []
        start = 0
        for c in counts:
            chunks.append(image_tokens[start:start + c])
            start += c
        return chunks

    def forward(
        self,
        images: List[Image.Image],
        image_counts: List[int],
        metrics: torch.Tensor | None = None,
        aux_images: List[Image.Image] | None = None,
        aux_image_counts: List[int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        image_tokens, image_grid_thw = self.encode_images(images)
        per_image_tokens = self._split_tokens_by_image(image_tokens, image_grid_thw)

        per_image_features = []
        for tok in per_image_tokens:
            feat = tok.mean(dim=0)
            per_image_features.append(feat)

        per_sample_features = []
        cursor = 0
        for n_img in image_counts:
            sample_feat = torch.stack(per_image_features[cursor: cursor + n_img], dim=0).mean(dim=0)
            per_sample_features.append(sample_feat)
            cursor += n_img

        x = torch.stack(per_sample_features, dim=0).float()

        if self.metrics_mlp is not None:
            if metrics is None:
                raise RuntimeError("metrics_mlp is enabled but metrics is None.")
            metrics = metrics.to(device=x.device, dtype=torch.float32)
            m = self.metrics_mlp(metrics)
            x = torch.cat([x, m.float()], dim=-1)

        x = x.to(device=self.norm.weight.device, dtype=self.norm.weight.dtype)
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return {"logits": logits, "features": x}


class TimmClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.1,
        img_size: int = 224,
        freeze_backbone: bool = False,
        num_metrics: int = 0,
        metrics_hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = int(self.backbone.num_features)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.img_size = int(img_size)
        self.num_metrics = int(num_metrics)
        self.metrics_hidden_dim = int(metrics_hidden_dim)

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        if self.num_metrics > 0:
            self.metrics_mlp = nn.Sequential(
                nn.Linear(self.num_metrics, self.metrics_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            final_dim = feat_dim + self.metrics_hidden_dim
        else:
            self.metrics_mlp = None
            final_dim = feat_dim

        self.norm = nn.LayerNorm(final_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_dim, num_classes)

        self.replaced_lora_modules: List[str] = []

    def _prepare_tensor(self, images: List[Image.Image], device: torch.device):
        xs = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            xs.append(self.transform(img))
        return torch.stack(xs, dim=0).to(device)

    def forward(
        self,
        images: List[Image.Image],
        image_counts: List[int],
        metrics: torch.Tensor | None = None,
        aux_images: List[Image.Image] | None = None,
        aux_image_counts: List[int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x = self._prepare_tensor(images, device=device)

        feats = self.backbone(x).float()

        per_sample_features = []
        cursor = 0
        for n_img in image_counts:
            sample_feat = feats[cursor: cursor + n_img].mean(dim=0)
            per_sample_features.append(sample_feat)
            cursor += n_img

        x = torch.stack(per_sample_features, dim=0)

        if self.metrics_mlp is not None:
            if metrics is None:
                raise RuntimeError("metrics_mlp is enabled but metrics is None.")
            metrics = metrics.to(device=device, dtype=torch.float32)
            m = self.metrics_mlp(metrics)
            x = torch.cat([x.float(), m.float()], dim=-1)

        x = x.to(device=self.norm.weight.device, dtype=self.norm.weight.dtype)
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return {"logits": logits, "features": x}


# class DualBranchTimmClassifier(nn.Module):
#     def __init__(
#         self,
#         image_backbone_name: str,
#         mask_backbone_name: str,
#         num_classes: int,
#         pretrained: bool = True,
#         dropout: float = 0.2,
#         img_size: int = 224,
#         freeze_image_backbone: bool = False,
#         freeze_mask_backbone: bool = False,
#         num_metrics: int = 0,
#         metrics_hidden_dim: int = 128,
#         image_feat_dim: int = 512,
#         mask_feat_dim: int = 256,
#     ):
#         super().__init__()

#         self.image_backbone = timm.create_model(
#             image_backbone_name,
#             pretrained=pretrained,
#             num_classes=0,
#             global_pool="avg",
#         )
#         self.mask_backbone = timm.create_model(
#             mask_backbone_name,
#             pretrained=pretrained,
#             num_classes=0,
#             global_pool="avg",
#         )

#         if freeze_image_backbone:
#             for p in self.image_backbone.parameters():
#                 p.requires_grad = False
#         if freeze_mask_backbone:
#             for p in self.mask_backbone.parameters():
#                 p.requires_grad = False

#         img_in_dim = int(self.image_backbone.num_features)
#         mask_in_dim = int(self.mask_backbone.num_features)

#         self.image_proj = nn.Sequential(
#             nn.Linear(img_in_dim, image_feat_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#         )
#         self.mask_proj = nn.Sequential(
#             nn.Linear(mask_in_dim, mask_feat_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#         )

#         self.num_metrics = int(num_metrics)
#         self.metrics_hidden_dim = int(metrics_hidden_dim)
#         if self.num_metrics > 0:
#             self.metrics_mlp = nn.Sequential(
#                 nn.Linear(self.num_metrics, self.metrics_hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             )
#             tab_dim = self.metrics_hidden_dim
#         else:
#             self.metrics_mlp = None
#             tab_dim = 0

#         final_dim = image_feat_dim + mask_feat_dim + tab_dim

#         self.norm = nn.LayerNorm(final_dim)
#         self.dropout = nn.Dropout(dropout)
#         from .mamba_fusion import MambaFusionHead

#         # 假设 img_feat_dim=512, mask_feat_dim=64, metrics_hidden_dim=128, num_classes=7
#      # 修改为
#         self.classifier = MambaFusionHead(
#             img_dim=image_feat_dim,
#             mask_dim=mask_feat_dim,
#             metrics_dim=metrics_hidden_dim,
#             token_dim=128,
#             num_classes=num_classes
#         )

#         self.img_size = int(img_size)
#         self.transform = T.Compose([
#             T.Resize((self.img_size, self.img_size), antialias=True),
#             T.ToTensor(),
#             T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ])

#         self.replaced_lora_modules: List[str] = []

#     def _prepare_tensor(self, images: List[Image.Image], device: torch.device):
#         xs = []
#         for img in images:
#             if img.mode != "RGB":
#                 img = img.convert("RGB")
#             xs.append(self.transform(img))
#         return torch.stack(xs, dim=0).to(device)

#     def _pool_per_sample(self, feats: torch.Tensor, counts: List[int]) -> torch.Tensor:
#         pooled = []
#         cursor = 0
#         for n_img in counts:
#             pooled.append(feats[cursor: cursor + n_img].mean(dim=0))
#             cursor += n_img
#         return torch.stack(pooled, dim=0)

#     def forward(
#         self,
#         images: List[Image.Image],
#         image_counts: List[int],
#         metrics: torch.Tensor | None = None,
#         aux_images: List[Image.Image] | None = None,
#         aux_image_counts: List[int] | None = None,
#     ) -> Dict[str, torch.Tensor]:
#         if aux_images is None or aux_image_counts is None:
#             raise RuntimeError("DualBranchTimmClassifier requires aux_images and aux_image_counts.")

#         device = next(self.parameters()).device

#         img_x = self._prepare_tensor(images, device)
#         mask_x = self._prepare_tensor(aux_images, device)

#         img_feats = self.image_backbone(img_x).float()
#         mask_feats = self.mask_backbone(mask_x).float()

#         img_feats = self._pool_per_sample(img_feats, image_counts)
#         mask_feats = self._pool_per_sample(mask_feats, aux_image_counts)

#         img_feats = self.image_proj(img_feats)
#         mask_feats = self.mask_proj(mask_feats)

#         fused = [img_feats, mask_feats]

#         if self.metrics_mlp is not None:
#             if metrics is None:
#                 raise RuntimeError("metrics_mlp is enabled but metrics is None.")
#             metrics = metrics.to(device=device, dtype=torch.float32)
#             tab = self.metrics_mlp(metrics)
#             fused.append(tab)

#         x = torch.cat(fused, dim=-1)
#         x = x.to(device=self.norm.weight.device, dtype=self.norm.weight.dtype)

#         x = self.norm(x)
#         x = self.dropout(x)
#         logits = self.classifier(x)

#         return {"logits": logits, "features": x}
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from PIL import Image
# import timm
# from .mamba_fusion import MambaFusionHead

class DualBranchTimmClassifier(nn.Module):
    def __init__(
        self,
        image_backbone_name: str,
        mask_backbone_name: str,
        num_classes: int,
        pretrained=True,
        dropout=0.2,
        img_size=224,
        freeze_image_backbone=False,
        freeze_mask_backbone=False,
        num_metrics=0,
        metrics_hidden_dim=128,
        image_feat_dim=512,
        mask_feat_dim=64,
    ):
        super().__init__()
        # backbones
        self.image_backbone = timm.create_model(
            image_backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.mask_backbone = timm.create_model(
            mask_backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        if freeze_image_backbone:
            for p in self.image_backbone.parameters(): p.requires_grad = False
        if freeze_mask_backbone:
            for p in self.mask_backbone.parameters(): p.requires_grad = False

        # 线性投影
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_backbone.num_features, image_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mask_proj = nn.Sequential(
            nn.Linear(self.mask_backbone.num_features, mask_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # metrics
        self.num_metrics = num_metrics
        self.metrics_hidden_dim = metrics_hidden_dim
        if num_metrics > 0:
            self.metrics_mlp = nn.Sequential(
                nn.Linear(num_metrics, metrics_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.metrics_mlp = None

        # Mamba 融合头
        self.classifier = MambaFusionHead(
            img_dim=image_feat_dim,
            mask_dim=mask_feat_dim if mask_backbone_name else None,
            metrics_dim=metrics_hidden_dim if num_metrics > 0 else None,
            token_dim=128,
            num_classes=num_classes
        )

        self.norm = nn.LayerNorm(image_feat_dim + mask_feat_dim + (metrics_hidden_dim if num_metrics > 0 else 0))
        self.dropout = nn.Dropout(dropout)

        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

    def _prepare_tensor(self, images, device):
        xs = []
        for img in images:
            if img.mode != "RGB": img = img.convert("RGB")
            xs.append(self.transform(img))
        return torch.stack(xs, dim=0).to(device)

    def _pool_per_sample(self, feats, counts):
        pooled = []
        cursor = 0
        for n in counts:
            pooled.append(feats[cursor:cursor+n].mean(dim=0))
            cursor += n
        return torch.stack(pooled, dim=0)

    def forward(self, images, image_counts, aux_images=None, aux_image_counts=None, metrics=None):
        device = next(self.parameters()).device
        img_feats = self._pool_per_sample(
            self.image_backbone(self._prepare_tensor(images, device)), image_counts
        )
        img_feats = self.image_proj(img_feats)

        mask_feats = None
        if aux_images is not None:
            mask_feats = self._pool_per_sample(
                self.mask_backbone(self._prepare_tensor(aux_images, device)), aux_image_counts
            )
            mask_feats = self.mask_proj(mask_feats)

        metrics_feats = None
        if self.metrics_mlp is not None:
            if metrics is None: raise RuntimeError("metrics_mlp enabled but metrics is None")
            metrics_feats = self.metrics_mlp(metrics.to(device=device, dtype=torch.float32))

        # 分类
        x = torch.cat([img_feats] + ([mask_feats] if mask_feats is not None else []) + ([metrics_feats] if metrics_feats is not None else []), dim=-1)
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(img_feats, mask_feats, metrics_feats)
        return {"logits": logits, "features": x}

def build_model(cfg) -> nn.Module:
    model_type = str(cfg["model_type"]).lower()
    sample_mode = str(cfg.get("sample_mode", "image"))
    metrics_cols = cfg.get("metrics_cols", []) or []

    use_metrics = sample_mode in {"mask_metrics", "image_metrics", "image_mask_metrics"}
    num_metrics = len(metrics_cols) if use_metrics else 0
    metrics_hidden_dim = int(cfg.get("metrics_hidden_dim", 128))

    if model_type == "qwen2vl":
        return Qwen2VLImageClassifier(
            model_name_or_path=cfg["model_name_or_path"],
            num_classes=int(cfg["num_classes"]),
            dropout=float(cfg["dropout"]),
            min_pixels=int(cfg["min_pixels"]),
            max_pixels=int(cfg["max_pixels"]),
            freeze_visual=bool(cfg["freeze_visual"]),
            use_lora=bool(cfg["use_lora"]),
            lora_r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            lora_targets=tuple(cfg.get("lora_targets", [])),
            num_metrics=num_metrics,
            metrics_hidden_dim=metrics_hidden_dim,
        )

    if model_type == "timm":
        return TimmClassifier(
            backbone_name=cfg["backbone_name"],
            num_classes=int(cfg["num_classes"]),
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg["dropout"]),
            img_size=int(cfg.get("img_size", 224)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            num_metrics=num_metrics,
            metrics_hidden_dim=metrics_hidden_dim,
        )

    if model_type == "dual_timm":
        return DualBranchTimmClassifier(
            image_backbone_name=cfg["image_backbone_name"],
            mask_backbone_name=cfg["mask_backbone_name"],
            num_classes=int(cfg["num_classes"]),
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg["dropout"]),
            img_size=int(cfg.get("img_size", 224)),
            freeze_image_backbone=bool(cfg.get("freeze_image_backbone", False)),
            freeze_mask_backbone=bool(cfg.get("freeze_mask_backbone", False)),
            num_metrics=num_metrics,
            metrics_hidden_dim=metrics_hidden_dim,
            image_feat_dim=int(cfg.get("image_feat_dim", 512)),
            mask_feat_dim=int(cfg.get("mask_feat_dim", 256)),
        )

    raise ValueError(f"Unsupported model_type: {model_type}")