import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int = 2, ff_dim: int = 128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, token_dim)
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class MambaFusionHead(nn.Module):
    def __init__(self, img_dim, mask_dim=None, metrics_dim=None, token_dim=128, num_classes=7):
        super().__init__()
        # 自动投影
        self.img_proj = nn.Linear(img_dim, token_dim)
        self.mask_proj = nn.Linear(mask_dim, token_dim) if mask_dim is not None else None
        self.metrics_proj = nn.Linear(metrics_dim, token_dim) if metrics_dim is not None else None

        self.mamba = MambaBlock(token_dim=token_dim, num_heads=2, ff_dim=token_dim*2)
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, img_feat, mask_feat=None, metrics_feat=None):
        tokens = []
        tokens.append(self.img_proj(img_feat).unsqueeze(1))
        if mask_feat is not None:
            tokens.append(self.mask_proj(mask_feat).unsqueeze(1))
        if metrics_feat is not None:
            tokens.append(self.metrics_proj(metrics_feat).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        x = self.mamba(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits