import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.ops import deform_conv2d

class FAF(nn.Module):
    """ Feature Adaptive Filter (FAF) """
    def __init__(self, in_channels):
        super(FAF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        # 官方实现中，FAF通常不改变通道数，或者在这里做降维
        # 为了匹配后续解码器输入，这里设定输出通道为 256 (原特征的一半)
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        return self.conv_reduce(x)

class FADB(nn.Module):
    """ Feature Alignment Decoder Block (FADB) """
    def __init__(self, in_channels, out_channels, flops_test=False):
        super(FADB, self).__init__()
        self.flops_test = flops_test
        
        # 偏移生成器: 输入是 [Up, Skip]
        # Up: in_channels, Skip: in_channels -> total 2*in
        self.conv_offset = nn.Conv2d(in_channels * 2, 18, 3, padding=1)
        self.deform_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 融合层
        self.conv_final = nn.Conv2d(in_channels + out_channels, out_channels, 1)

    def forward(self, x_up, x_skip):
        # 尺寸对齐
        if x_up.size(2) != x_skip.size(2):
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=True)
        
        x_cat = torch.cat([x_up, x_skip], dim=1)
        
        if self.flops_test:
            # FLOPs测试时用普通卷积代替，避免thop无法追踪deform_conv2d
            x_aligned = self.deform_conv(x_skip)
        else:
            offset = self.conv_offset(x_cat)
            x_aligned = deform_conv2d(x_skip, offset, self.deform_conv.weight, self.deform_conv.bias, padding=1)
            
        x_aligned = self.relu(self.bn(x_aligned))
        x_fuse = torch.cat([x_up, x_aligned], dim=1)
        return self.conv_final(x_fuse)

class MSFM(nn.Module):
    """ Multi-scale Soft Fusion Module (MSFM) """
    def __init__(self, in_channels, out_channels):
        super(MSFM, self).__init__()
        self.dconv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.dconv2 = nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3)
        self.dconv3 = nn.Conv2d(in_channels, out_channels, 3, padding=5, dilation=5)
        self.sam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )
        self.conv_fuse = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(x)
        x3 = self.dconv3(x)
        att1, att2, att3 = self.sam(x1), self.sam(x2), self.sam(x3)
        x_cat = torch.cat([x1*att1, x2*att2, x3*att3], dim=1)
        return self.conv_fuse(x_cat)

class JointSeg_Official(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, flops_test=False):
        super(JointSeg_Official, self).__init__()
        self.flops_test = flops_test
        
        # 1. Encoder: Standard VGG16
        # 使用 weights=None 避免警告
        vgg = vgg16(weights=None)
        
        # 适配输入通道
        if input_channels != 3:
            vgg.features[0] = nn.Conv2d(input_channels, 64, 3, padding=1)
            
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[:5])    # -> 64
        self.enc2 = nn.Sequential(*features[5:10])  # -> 128
        self.enc3 = nn.Sequential(*features[10:17]) # -> 256
        self.enc4 = nn.Sequential(*features[17:24]) # -> 512
        self.enc5 = nn.Sequential(*features[24:])   # -> 512
        
        # 【核心差异点】Heavy Center (模拟 VGG-16 全连接层转卷积)
        # 这是参数量达到 154M 的关键。
        # fc6 (4096) -> fc7 (4096)
        self.center = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3), # 7x7 卷积, params ~ 102M
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, 1),           # 1x1 卷积, params ~ 16M
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            # 降维以适配 Decoder
            nn.Conv2d(4096, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 2. FAF
        self.faf_faz = FAF(512) # 512 -> 256
        self.faf_rv = FAF(512)  # 512 -> 256

        # 3. FAZ Decoder
        # VGG5级下采样，这里逐级上采样
        self.faz_up5 = FADB(256, 256, flops_test) # In: 256, Skip: 512->256
        self.faz_up4 = FADB(256, 128, flops_test)
        self.faz_up3 = FADB(128, 64, flops_test)
        self.faz_up2 = FADB(64, 32, flops_test)
        self.faz_head = nn.Conv2d(32, num_classes, 1)

        # 4. RV Decoder
        self.rv_msfm1 = MSFM(256, 128)
        self.rv_msfm2 = MSFM(128, 64)
        self.rv_msfm3 = MSFM(64, 32)
        self.rv_msfm4 = MSFM(32, 16)
        self.rv_head = nn.Conv2d(16, num_classes, 1)
        
        # Skip Connection Reducers
        self.skip_reduce = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 64, 1) 
        ])

    def forward(self, x):
        # Encode
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        
        # Center (Heavy)
        feat = self.center(x5)
        
        # Filter
        f_faz = self.faf_faz(feat)
        f_rv = self.faf_rv(feat)
        
        # FAZ Decode
        x_skip = self.skip_reduce[0](x5)
        d5 = self.faz_up5(f_faz, x_skip)
        d5 = F.interpolate(d5, scale_factor=2, mode='bilinear')
        
        x_skip = self.skip_reduce[1](x4)
        d4 = self.faz_up4(d5, x_skip)
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear')
        
        x_skip = self.skip_reduce[2](x3)
        d3 = self.faz_up3(d4, x_skip)
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear')
        
        x_skip = self.skip_reduce[3](x2)
        d2 = self.faz_up2(d3, x_skip)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear')
        
        out_faz = self.faz_head(d2)
        out_faz = F.interpolate(out_faz, size=x.shape[2:], mode='bilinear')
        
        # RV Decode
        r5 = self.rv_msfm1(f_rv)
        r5 = F.interpolate(r5, scale_factor=2, mode='bilinear')
        r4 = self.rv_msfm2(r5)
        r4 = F.interpolate(r4, scale_factor=2, mode='bilinear')
        r3 = self.rv_msfm3(r4)
        r3 = F.interpolate(r3, scale_factor=2, mode='bilinear')
        r2 = self.rv_msfm4(r3)
        r2 = F.interpolate(r2, scale_factor=2, mode='bilinear')
        
        out_rv = self.rv_head(r2)
        out_rv = F.interpolate(out_rv, size=x.shape[2:], mode='bilinear')

        if self.flops_test:
            return out_faz
        return out_faz, out_rv