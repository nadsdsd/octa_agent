# models.py
# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# class Identity(nn.Module):
#     def forward(self, x): return x

# def _change_first_conv(model, in_ch: int):
#     # ResNet
#     if hasattr(model, 'conv1'):
#         old = model.conv1
#         model.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
#                                 stride=old.stride, padding=old.padding, bias=False)
#         return
#     # EfficientNet-B0
#     if hasattr(model, 'features') and isinstance(model.features[0][0], nn.Conv2d):
#         first = model.features[0][0]
#         model.features[0][0] = nn.Conv2d(in_ch, first.out_channels, kernel_size=first.kernel_size,
#                                          stride=first.stride, padding=first.padding, bias=False)
#         return
#     raise RuntimeError("未能修改首卷积，未知结构")

# class ViT2ch(nn.Module):
#     """ViT: 用 1x1 conv 将 2 通道 -> 3 通道，再送入 ViT；分类头支持拼接 metrics"""
#     def __init__(self, vit: nn.Module, num_classes: int, metrics_dim: int = 0, hid: int = 256):
#         super().__init__()
#         self.pre = nn.Conv2d(2, 3, kernel_size=1, bias=False)
#         self.vit = vit
#         feat_dim = vit.heads.head.in_features
#         vit.heads = Identity()
#         self.use_metrics = metrics_dim > 0
#         self.classifier = nn.Sequential(
#             nn.Linear(feat_dim + metrics_dim, hid),
#             nn.ReLU(inplace=True),
#             nn.Linear(hid, num_classes),
#         )
#     def forward(self, x, metrics: Optional[torch.Tensor] = None):
#         x = self.pre(x)
#         feat = self.vit(x)  # (B, feat_dim)
#         if self.use_metrics and metrics is not None:
#             feat = torch.cat([feat, metrics], dim=1)
#         return self.classifier(feat)

# def build_model(arch: str, num_classes: int, use_metrics: bool = False, metrics_dim: int = 0):
#     arch = arch.lower()
#     if arch in ["resnet50","resnet-50"]:
#         net = models.resnet50(weights=None)
#         _change_first_conv(net, in_ch=2)
#         feat_dim = net.fc.in_features
#         net.fc = Identity()
#         head_in = feat_dim + (metrics_dim if use_metrics else 0)
#         head = nn.Sequential(
#             nn.Linear(head_in, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, num_classes),
#         )
#         class Wrapper(nn.Module):
#             def __init__(self, backbone, head, use_metrics):
#                 super().__init__()
#                 self.backbone = backbone
#                 self.head = head
#                 self.use_metrics = use_metrics
#             def forward(self, x, metrics=None):
#                 feat = self.backbone.conv1(x)
#                 feat = self.backbone.bn1(feat)
#                 feat = self.backbone.relu(feat)
#                 feat = self.backbone.maxpool(feat)
#                 feat = self.backbone.layer1(feat)
#                 feat = self.backbone.layer2(feat)
#                 feat = self.backbone.layer3(feat)
#                 feat = self.backbone.layer4(feat)
#                 feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
#                 if self.use_metrics and metrics is not None:
#                     feat = torch.cat([feat, metrics], dim=1)
#                 return self.head(feat)
#         return Wrapper(net, head, use_metrics)

#     # if arch in ["efficientnet_b0","efficientnet-b0","efficientnet"]:
#     #     net = models.efficientnet_b0(weights=None)
#     #     _change_first_conv(net, in_ch=2)
#     #     feat_dim = net.classifier[1].in_features
#     #     net.classifier = nn.Identity()
#     #     class Wrapper(nn.Module):
#     #         def __init__(self, backbone, num_classes, use_metrics, mdim):
#     #             super().__init__()
#     #             self.backbone = backbone
#     #             self.use_metrics = use_metrics
#     #             self.head = nn.Sequential(
#     #                 nn.Linear(feat_dim + (mdim if use_metrics else 0), 256),
#     #                 nn.ReLU(inplace=True),
#     #                 nn.Linear(256, num_classes),
#     #             )
#     #         def forward(self, x, metrics=None):
#     #             feat = self.backbone(x)                      # (B,C,H,W)
#     #             feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
#     #             if self.use_metrics and metrics is not None:
#     #                 feat = torch.cat([feat, metrics], dim=1)
#     #             return self.head(feat)
#     #     return Wrapper(net, num_classes, use_metrics, metrics_dim)
#     # models.py (仅替换 EfficientNet 这段)
#     elif arch in ["efficientnet_b0","efficientnet-b0","efficientnet"]:
#         net = models.efficientnet_b0(weights=None)
#         _change_first_conv(net, in_ch=2)
#         feat_dim = net.classifier[1].in_features  # 1280
#         # 不用原 classifier；avgpool 用我们自己的
#         net.classifier = nn.Identity()

#         class Wrapper(nn.Module):
#             def __init__(self, backbone, num_classes, use_metrics, mdim):
#                 super().__init__()
#                 self.backbone = backbone
#                 self.use_metrics = use_metrics
#                 self.head = nn.Sequential(
#                     nn.Linear(feat_dim + (mdim if use_metrics else 0), 256),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(256, num_classes),
#                 )

#             def forward(self, x, metrics=None):
#                 # 正确的特征提取方式：features -> GAP -> flatten
#                 feat = self.backbone.features(x)               # (B, 1280, H/32, W/32)
#                 feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 1280)
#                 if self.use_metrics and metrics is not None:
#                     feat = torch.cat([feat, metrics], dim=1)
#                 return self.head(feat)

#         return Wrapper(net, num_classes, use_metrics, metrics_dim)


#     if arch in ["vit_b_16","vit-b16","vit"]:
#         vit = models.vit_b_16(weights=None)
#         return ViT2ch(vit, num_classes=num_classes, metrics_dim=(metrics_dim if use_metrics else 0))

#     raise ValueError(f"未知架构: {arch}")
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------------------------------------------------------------------
# #-*- coding: utf-8 -*-
# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# import numpy as np
# from sklearn.feature_extraction.text import HashingVectorizer

# # ======= 公共 =======

# class Identity(nn.Module):
#     def forward(self, x): return x

# def _change_first_conv_to_2ch(model):
#     """把首层卷积改为 2 通道（ResNet/EfficientNet 常见结构）"""
#     if hasattr(model, 'conv1'):
#         old = model.conv1
#         model.conv1 = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
#                                 stride=old.stride, padding=old.padding, bias=False)
#         return True
#     if hasattr(model, 'features') and isinstance(model.features[0][0], nn.Conv2d):
#         first = model.features[0][0]
#         model.features[0][0] = nn.Conv2d(2, first.out_channels, kernel_size=first.kernel_size,
#                                          stride=first.stride, padding=first.padding, bias=False)
#         return True
#     return False

# # ======= 仅图像 / 图像+数值 =======

# class ViT2ch(nn.Module):
#     """ViT：1×1 卷积把 2ch->3ch，再送 ViT；分类头可拼 metrics"""
#     def __init__(self, vit: nn.Module, num_classes: int, metrics_dim: int = 0, hid: int = 256):
#         super().__init__()
#         self.pre = nn.Conv2d(2, 3, kernel_size=1, bias=False)
#         self.vit = vit
#         feat_dim = vit.heads.head.in_features
#         vit.heads = Identity()
#         self.use_metrics = metrics_dim > 0
#         self.classifier = nn.Sequential(
#             nn.Linear(feat_dim + metrics_dim, hid),
#             nn.ReLU(inplace=True),
#             nn.Linear(hid, num_classes),
#         )
#     def forward(self, x, metrics: Optional[torch.Tensor] = None):
#         x = self.pre(x)
#         feat = self.vit(x)  # (B, feat_dim)
#         if self.use_metrics and metrics is not None:
#             feat = torch.cat([feat, metrics], dim=1)
#         return self.classifier(feat)

# def build_model(arch: str, num_classes: int, use_metrics: bool = False, metrics_dim: int = 0):
#     arch = arch.lower()
#     if arch in ["resnet50","resnet-50"]:
#         net = models.resnet50(weights=None)
#         ok = _change_first_conv_to_2ch(net)
#         if not ok:
#             raise RuntimeError("无法把 ResNet50 首层改为 2 通道")
#         feat_dim = net.fc.in_features
#         net.fc = Identity()
#         head_in = feat_dim + (metrics_dim if use_metrics else 0)
#         head = nn.Sequential(
#             nn.Linear(head_in, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, num_classes),
#         )
#         class Wrapper(nn.Module):
#             def __init__(self, backbone, head, use_metrics):
#                 super().__init__()
#                 self.backbone = backbone
#                 self.head = head
#                 self.use_metrics = use_metrics
#             def forward(self, x, metrics=None):
#                 m = self.backbone
#                 feat = m.conv1(x); feat = m.bn1(feat); feat = m.relu(feat); feat = m.maxpool(feat)
#                 feat = m.layer1(feat); feat = m.layer2(feat); feat = m.layer3(feat); feat = m.layer4(feat)
#                 feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
#                 if self.use_metrics and metrics is not None:
#                     feat = torch.cat([feat, metrics], dim=1)
#                 return self.head(feat)
#         return Wrapper(net, head, use_metrics)

#     if arch in ["efficientnet_b0","efficientnet-b0","efficientnet"]:
#         net = models.efficientnet_b0(weights=None)
#         ok = _change_first_conv_to_2ch(net)
#         if not ok:
#             raise RuntimeError("无法把 EfficientNet-B0 首层改为 2 通道")
#         feat_dim = net.classifier[1].in_features  # 1280
#         net.classifier = nn.Identity()
#         class Wrapper(nn.Module):
#             def __init__(self, backbone, num_classes, use_metrics, mdim):
#                 super().__init__()
#                 self.backbone = backbone
#                 self.use_metrics = use_metrics
#                 self.head = nn.Sequential(
#                     nn.Linear(feat_dim + (mdim if use_metrics else 0), 256),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(256, num_classes),
#                 )
#             def forward(self, x, metrics=None):
#                 m = self.backbone
#                 feat = m.features(x)
#                 feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
#                 if self.use_metrics and metrics is not None:
#                     feat = torch.cat([feat, metrics], dim=1)
#                 return self.head(feat)
#         return Wrapper(net, num_classes, use_metrics, metrics_dim)

#     if arch in ["vit_b_16","vit-b16","vit"]:
#         vit = models.vit_b_16(weights=None)
#         return ViT2ch(vit, num_classes=num_classes, metrics_dim=(metrics_dim if use_metrics else 0))

#     raise ValueError(f"未知架构: {arch}")

# # ======= 文本（纯本地 HashingVectorizer） + 图像 融合 =======

# class OfflineTextEncoder(nn.Module):
#     """
#     纯本地文本编码（无词典/无训练/无需联网）：
#     HashingVectorizer -> 稀疏 TF 特征 -> dense float32 tensor
#     """
#     def __init__(self, n_features: int = 4096):
#         super().__init__()
#         self.n_features = n_features
#         self._hv = HashingVectorizer(
#             n_features=n_features,
#             alternate_sign=False,
#             norm="l2",
#             lowercase=True,
#             analyzer="word",
#             ngram_range=(1, 2),
#         )
#         self.out_dim = n_features

#     def forward(self, raw_texts: list) -> torch.Tensor:
#         X = self._hv.transform(raw_texts)   # scipy CSR
#         X = X.astype(np.float32)
#         dense = torch.from_numpy(X.toarray())  # [B, n_features]
#         return dense

# class ImgBackbone(nn.Module):
#     """统一图像特征抽取为 GAP 向量"""
#     def __init__(self, arch: str):
#         super().__init__()
#         self.arch = arch.lower()
#         if self.arch in ["resnet50","resnet-50"]:
#             net = models.resnet50(weights=None)
#             if not _change_first_conv_to_2ch(net):
#                 raise RuntimeError("ResNet50 首层 2ch 失败")
#             self.backbone = net
#             self.feat_dim = net.fc.in_features
#             net.fc = nn.Identity()
#         elif self.arch in ["efficientnet_b0","efficientnet-b0","efficientnet"]:
#             net = models.efficientnet_b0(weights=None)
#             if not _change_first_conv_to_2ch(net):
#                 raise RuntimeError("EfficientNet-B0 首层 2ch 失败")
#             self.backbone = net
#             self.feat_dim = net.classifier[1].in_features
#             net.classifier = nn.Identity()
#         elif self.arch in ["vit_b_16","vit-b16","vit"]:
#             vit = models.vit_b_16(weights=None)
#             self.pre = nn.Conv2d(2, 3, kernel_size=1, bias=False)
#             self.backbone = vit
#             self.feat_dim = vit.heads.head.in_features
#             vit.heads = nn.Identity()
#         else:
#             raise ValueError(f"未知架构: {arch}")

#     def forward(self, x):
#         a = self.arch
#         if a.startswith("vit"):
#             x = self.pre(x)
#             feat = self.backbone(x)                        # [B, D]
#             return feat
#         if a.startswith("resnet"):
#             m = self.backbone
#             feat = m.conv1(x); feat = m.bn1(feat); feat = m.relu(feat); feat = m.maxpool(feat)
#             feat = m.layer1(feat); feat = m.layer2(feat); feat = m.layer3(feat); feat = m.layer4(feat)
#             feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
#             return feat
#         # efficientnet
#         m = self.backbone
#         feat = m.features(x)
#         feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
#         return feat

# class MultiModalImgTextModel(nn.Module):
#     """图像特征 + 文本特征（本地 Hashing）拼接 -> MLP 分类"""
#     def __init__(self, arch: str, num_classes: int, proj_dim: int = 512, drop: float = 0.2, text_dim: int = 4096):
#         super().__init__()
#         self.img = ImgBackbone(arch)
#         self.txt = OfflineTextEncoder(n_features=text_dim)
#         self.img_proj = nn.Linear(self.img.feat_dim, proj_dim)
#         self.txt_proj = nn.Linear(self.txt.out_dim, proj_dim)
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(proj_dim * 2),
#             nn.Linear(proj_dim * 2, proj_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(proj_dim, num_classes),
#         )
#     def forward(self, x_img, raw_texts: list):
#         img_feat = self.img(x_img)                    # [B, Di]
#         txt_feat = self.txt(raw_texts)                # [B, Dt]（CPU tensor）
#         if txt_feat.device != img_feat.device:
#             txt_feat = txt_feat.to(img_feat.device)
#         img_feat = self.img_proj(img_feat)            # [B, P]
#         txt_feat = self.txt_proj(txt_feat)            # [B, P]
#         fused = torch.cat([img_feat, txt_feat], dim=1)
#         return self.classifier(fused)

# # 供 train.py 调用
# def build_model_multimodal_text(arch: str, num_classes: int):
#     return MultiModalImgTextModel(arch=arch, num_classes=num_classes)
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ===================== RVMamba 导入检查 =====================
# 尝试导入 RVMamba，如果失败（通常是因为缺少 HDFE.py/VMAF.py 等依赖），则记录错误
try:
    from RVMamba import RVMamba
except ImportError as e:
    print(f"\n{'!'*60}")
    print(f"!!! [models.py] 无法导入 RVMamba 模块 !!!")
    print(f"错误详情: {e}")
    print("请确保当前目录下包含 RVMamba.py 及其所有依赖文件：")
    print("HDFE.py, VMAF.py, wtconv2d.py, AdaptiveFeatureFusion.py, lsa.py")
    print(f"{'!'*60}\n")
    RVMamba = None

# ===================== 基础工具 =====================

class Identity(nn.Module):
    def forward(self, x): return x

def _change_first_conv_to_2ch(model):
    """把首层卷积改为 2 通道（ResNet/EfficientNet 常见结构）"""
    if hasattr(model, 'conv1'):
        old = model.conv1
        model.conv1 = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                                stride=old.stride, padding=old.padding, bias=False)
        return True
    if hasattr(model, 'features') and isinstance(model.features[0][0], nn.Conv2d):
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(2, first.out_channels, kernel_size=first.kernel_size,
                                         stride=first.stride, padding=first.padding, bias=False)
        return True
    return False

# ===================== RVMamba 适配器 =====================

class RVMambaWrapper(nn.Module):
    """
    [优化版] RVMamba 编码器模式：
    1. 修改首层以接受 2 通道输入 (RV + FAZ)。
    2. 仅运行 Encoder + Bottleneck，直接输出高维语义特征 (256维)。
    3. 移除了分类任务不需要的 Decoder，节省显存并加速。
    """
    def __init__(self, qseme_type='conservative'):
        super().__init__()
        if RVMamba is None:
            raise ImportError("RVMamba 模块未成功加载，无法创建 RVMambaWrapper。请检查控制台顶部的导入错误信息。")
        
        # 实例化完整模型
        full_model = RVMamba(qseme_type=qseme_type)
        
        # --- 1. 提取并保留编码器部分 ---
        self.qseme = full_model.qseme
        self.e1 = full_model.e1
        self.e2 = full_model.e2
        self.e3 = full_model.e3
        self.bottleneck = full_model.bottleneck
        
        # --- 2. 修改输入层为 2 通道 ---
        # 原始代码通常是 Conv2d(1, 32, ...)
        if hasattr(self.qseme, 'init_conv') and len(self.qseme.init_conv) > 0:
            old_conv = self.qseme.init_conv[0]
            if isinstance(old_conv, nn.Conv2d):
                self.qseme.init_conv[0] = nn.Conv2d(
                    2, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=(old_conv.bias is not None)
                )
        
        # --- 3. 设定特征维度 ---
        # RVMamba 的 bottleneck 输出维度是 256 (MediumChannelOCTAMambaBlock(128, 256))
        self.num_features = 256 

        # 注意：这里我们不再保留 full_model.d1, d2, d3，它们会被自动回收，节省显存

    def forward(self, x):
        # 1. QSEME / Input
        x = self.qseme(x)  # -> (B, 16, H, W)
        
        # 2. Encoder (下采样)
        x, _ = self.e1(x)    # 16->32,  H/2
        x, _ = self.e2(x)    # 32->64,  H/4
        x, _ = self.e3(x)    # 64->128, H/8
        
        # 3. Bottleneck (最高层语义)
        x = self.bottleneck(x)   # 128->256, H/8
        
        # 4. Global Average Pooling (Features)
        # 此时 x 的形状为 (B, 256, H/8, W/8) -> GAP -> (B, 256)
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return feat

# ===================== 模型构建逻辑 =====================

class ViT2ch(nn.Module):
    """ViT：1×1 卷积把 2ch->3ch，再送 ViT；分类头可拼 metrics"""
    def __init__(self, vit: nn.Module, num_classes: int, metrics_dim: int = 0, hid: int = 256):
        super().__init__()
        self.pre = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self.vit = vit
        feat_dim = vit.heads.head.in_features
        vit.heads = Identity()
        self.use_metrics = metrics_dim > 0
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + metrics_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, num_classes),
        )
    def forward(self, x, metrics: Optional[torch.Tensor] = None):
        x = self.pre(x)
        feat = self.vit(x)  # (B, feat_dim)
        if self.use_metrics and metrics is not None:
            feat = torch.cat([feat, metrics], dim=1)
        return self.classifier(feat)

def build_model(arch: str, num_classes: int, use_metrics: bool = False, metrics_dim: int = 0):
    arch = arch.lower()

    # --- RVMamba 分支 ---
    if arch == "rvmamba":
        backbone = RVMambaWrapper(qseme_type='conservative')
        feat_dim = backbone.num_features # 32
        
        head_in = feat_dim + (metrics_dim if use_metrics else 0)
        
        # 自定义分类头
        head = nn.Sequential(
            nn.Linear(head_in, 128), # RVMamba 维度较小，中间层设为 128
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        
        class Wrapper(nn.Module):
            def __init__(self, bb, hd, um):
                super().__init__()
                self.backbone = bb
                self.head = hd
                self.use_metrics = um
            def forward(self, x, metrics=None):
                feat = self.backbone(x) # (B, 32)
                if self.use_metrics and metrics is not None:
                    feat = torch.cat([feat, metrics], dim=1)
                return self.head(feat)
        return Wrapper(backbone, head, use_metrics)

    # --- ResNet50 ---
    if arch in ["resnet50","resnet-50"]:
        net = models.resnet50(weights=None)
        ok = _change_first_conv_to_2ch(net)
        if not ok:
            raise RuntimeError("无法把 ResNet50 首层改为 2 通道")
        feat_dim = net.fc.in_features
        net.fc = Identity()
        head_in = feat_dim + (metrics_dim if use_metrics else 0)
        head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        class Wrapper(nn.Module):
            def __init__(self, backbone, head, use_metrics):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.use_metrics = use_metrics
            def forward(self, x, metrics=None):
                m = self.backbone
                feat = m.conv1(x); feat = m.bn1(feat); feat = m.relu(feat); feat = m.maxpool(feat)
                feat = m.layer1(feat); feat = m.layer2(feat); feat = m.layer3(feat); feat = m.layer4(feat)
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                if self.use_metrics and metrics is not None:
                    feat = torch.cat([feat, metrics], dim=1)
                return self.head(feat)
        return Wrapper(net, head, use_metrics)

    # --- EfficientNet ---
    if arch in ["efficientnet_b0","efficientnet-b0","efficientnet"]:
        net = models.efficientnet_b0(weights=None)
        ok = _change_first_conv_to_2ch(net)
        if not ok:
            raise RuntimeError("无法把 EfficientNet-B0 首层改为 2 通道")
        feat_dim = net.classifier[1].in_features  # 1280
        net.classifier = nn.Identity()
        class Wrapper(nn.Module):
            def __init__(self, backbone, num_classes, use_metrics, mdim):
                super().__init__()
                self.backbone = backbone
                self.use_metrics = use_metrics
                self.head = nn.Sequential(
                    nn.Linear(feat_dim + (mdim if use_metrics else 0), 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, num_classes),
                )
            def forward(self, x, metrics=None):
                m = self.backbone
                feat = m.features(x)
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                if self.use_metrics and metrics is not None:
                    feat = torch.cat([feat, metrics], dim=1)
                return self.head(feat)
        return Wrapper(net, num_classes, use_metrics, metrics_dim)

    # --- ViT ---
    if arch in ["vit_b_16","vit-b16","vit"]:
        vit = models.vit_b_16(weights=None)
        return ViT2ch(vit, num_classes=num_classes, metrics_dim=(metrics_dim if use_metrics else 0))

    raise ValueError(f"未知架构: {arch}")

# ===================== 多模态文本 + 图像 融合 =====================

class OfflineTextEncoder(nn.Module):
    """
    纯本地文本编码（无词典/无训练/无需联网）：
    HashingVectorizer -> 稀疏 TF 特征 -> dense float32 tensor
    """
    def __init__(self, n_features: int = 4096):
        super().__init__()
        self.n_features = n_features
        self._hv = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
        )
        self.out_dim = n_features

    def forward(self, raw_texts: list) -> torch.Tensor:
        X = self._hv.transform(raw_texts)   # scipy CSR
        X = X.astype(np.float32)
        dense = torch.from_numpy(X.toarray())  # [B, n_features]
        return dense

class ImgBackbone(nn.Module):
    """统一图像特征抽取为 GAP 向量"""
    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch.lower()
        
        if self.arch == "rvmamba":
            self.backbone = RVMambaWrapper(qseme_type='conservative')
            self.feat_dim = self.backbone.num_features # 32
            
        elif self.arch in ["resnet50","resnet-50"]:
            net = models.resnet50(weights=None)
            if not _change_first_conv_to_2ch(net):
                raise RuntimeError("ResNet50 首层 2ch 失败")
            self.backbone = net
            self.feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif self.arch in ["efficientnet_b0","efficientnet-b0","efficientnet"]:
            net = models.efficientnet_b0(weights=None)
            if not _change_first_conv_to_2ch(net):
                raise RuntimeError("EfficientNet-B0 首层 2ch 失败")
            self.backbone = net
            self.feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif self.arch in ["vit_b_16","vit-b16","vit"]:
            vit = models.vit_b_16(weights=None)
            self.pre = nn.Conv2d(2, 3, kernel_size=1, bias=False)
            self.backbone = vit
            self.feat_dim = vit.heads.head.in_features
            vit.heads = nn.Identity()
        else:
            raise ValueError(f"未知架构: {arch}")

    def forward(self, x):
        if self.arch == "rvmamba":
            return self.backbone(x) # RVMambaWrapper 直接返回 GAP 特征

        a = self.arch
        if a.startswith("vit"):
            x = self.pre(x)
            feat = self.backbone(x)                        # [B, D]
            return feat
        if a.startswith("resnet"):
            m = self.backbone
            feat = m.conv1(x); feat = m.bn1(feat); feat = m.relu(feat); feat = m.maxpool(feat)
            feat = m.layer1(feat); feat = m.layer2(feat); feat = m.layer3(feat); feat = m.layer4(feat)
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return feat
        # efficientnet
        m = self.backbone
        feat = m.features(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return feat

class MultiModalImgTextModel(nn.Module):
    """图像特征 + 文本特征（本地 Hashing）拼接 -> MLP 分类"""
    def __init__(self, arch: str, num_classes: int, proj_dim: int = 512, drop: float = 0.2, text_dim: int = 4096):
        super().__init__()
        self.img = ImgBackbone(arch)
        self.txt = OfflineTextEncoder(n_features=text_dim)
        self.img_proj = nn.Linear(self.img.feat_dim, proj_dim)
        self.txt_proj = nn.Linear(self.txt.out_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(proj_dim, num_classes),
        )
    def forward(self, x_img, raw_texts: list):
        img_feat = self.img(x_img)                    # [B, Di]
        txt_feat = self.txt(raw_texts)                # [B, Dt]（CPU tensor）
        if txt_feat.device != img_feat.device:
            txt_feat = txt_feat.to(img_feat.device)
        img_feat = self.img_proj(img_feat)            # [B, P]
        txt_feat = self.txt_proj(txt_feat)            # [B, P]
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)

# 供 train.py 调用
def build_model_multimodal_text(arch: str, num_classes: int):
    return MultiModalImgTextModel(arch=arch, num_classes=num_classes)