from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear only supports nn.Linear base layers.")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.r <= 0:
            return base_out

        x_lora = self.dropout(x)
        delta = F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta


def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_by_suffix(
    root: nn.Module,
    target_suffixes: Iterable[str],
    r: int = 4,
    alpha: int = 16,
    dropout: float = 0.05,
) -> List[str]:
    target_suffixes = tuple(target_suffixes)
    replaced = []

    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue

        parent, child_name = _get_parent_module(root, name)
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)

    return replaced


def get_lora_state_dict(model: nn.Module):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.r > 0:
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict):
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.r > 0:
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state_dict and key_b in state_dict:
                module.lora_A.data.copy_(state_dict[key_a])
                module.lora_B.data.copy_(state_dict[key_b])
            else:
                missing.append(name)
    return missing