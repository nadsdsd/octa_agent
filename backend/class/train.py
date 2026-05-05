from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.dataset import ClassificationCollator, OCTA500ClassificationDataset
from src.losses import build_loss
from src.lora import get_lora_state_dict
from src.metrics import compute_metrics
from src.model import build_model
from src.utils import count_trainable_parameters, load_yaml, save_json, set_seed


class EarlyStopper:
    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def get_head_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        "norm.weight": model.norm.weight.detach().cpu(),
        "norm.bias": model.norm.bias.detach().cpu(),
        "classifier.weight": model.classifier.weight.detach().cpu(),
        "classifier.bias": model.classifier.bias.detach().cpu(),
    }


def build_checkpoint(model, cfg, label_names, epoch, best_metric_name, param_info, metrics_mean, metrics_std):
    model_type = str(cfg["model_type"]).lower()

    if model_type == "qwen2vl":
        ckpt = {
            "model_type": model_type,
            "head_state_dict": get_head_state_dict(model),
            "lora_state_dict": get_lora_state_dict(model) if bool(cfg.get("use_lora", False)) else {},
            "config": cfg,
            "label_names": label_names,
            "epoch": epoch,
            "best_metric_name": best_metric_name,
            "param_info": param_info,
            "save_type": "head_plus_lora" if bool(cfg.get("use_lora", False)) else "head_only",
            "metrics_mean": metrics_mean.tolist() if metrics_mean is not None else None,
            "metrics_std": metrics_std.tolist() if metrics_std is not None else None,
        }
    else:
        ckpt = {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "label_names": label_names,
            "epoch": epoch,
            "best_metric_name": best_metric_name,
            "param_info": param_info,
            "save_type": "full_model",
            "metrics_mean": metrics_mean.tolist() if metrics_mean is not None else None,
            "metrics_std": metrics_std.tolist() if metrics_std is not None else None,
        }

    return ckpt


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    label_names: List[str],
    use_fp16: bool,
    loss_fn,
) -> Dict:
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0
    n = 0

    amp_enabled = use_fp16 and torch.cuda.is_available()

    for batch in tqdm(loader, desc="eval", leave=False):
        labels = torch.tensor(batch.labels, dtype=torch.long, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(
                images=batch.images,
                image_counts=batch.image_counts,
                metrics=batch.metrics,
                aux_images=batch.aux_images,
                aux_image_counts=batch.aux_image_counts,
            )
            logits = outputs["logits"]

        loss = loss_fn(logits.float(), labels)

        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)

        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, label_names)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def compute_metrics_norm(df: pd.DataFrame, metrics_cols: List[str]):
    if not metrics_cols:
        return None, None
    x = df[metrics_cols].astype(float).to_numpy(dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_weighted_sampler(labels, num_classes: int, power: float = 1.0):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts <= 0, 1.0, counts)
    class_weights = 1.0 / np.power(counts, power)
    sample_weights = class_weights[np.array(labels, dtype=np.int64)]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_weights


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_mode = str(cfg.get("sample_mode", "image"))
    metrics_cols = cfg.get("metrics_cols", []) or []

    train_df = pd.read_csv(cfg["train_csv"])
    metrics_mean, metrics_std = compute_metrics_norm(train_df, metrics_cols)

    train_ds = OCTA500ClassificationDataset(
        cfg["train_csv"],
        cfg["image_cols"],
        cfg["label_col"],
        cfg["label_name_col"],
        sample_mode=sample_mode,
        metrics_cols=metrics_cols,
        mask_size=int(cfg.get("mask_size", 224)),
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        aux_mode=str(cfg.get("aux_mode", "fusion")),
    )
    val_ds = OCTA500ClassificationDataset(
        cfg["val_csv"],
        cfg["image_cols"],
        cfg["label_col"],
        cfg["label_name_col"],
        sample_mode=sample_mode,
        metrics_cols=metrics_cols,
        mask_size=int(cfg.get("mask_size", 224)),
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        aux_mode=str(cfg.get("aux_mode", "fusion")),
    )

    label_map_df = pd.concat([train_ds.df, val_ds.df], axis=0, ignore_index=True)
    label_names = [
        x for _, x in sorted(
            label_map_df[[cfg["label_col"], cfg["label_name_col"]]]
            .drop_duplicates()
            .values.tolist()
        )
    ]

    collator = ClassificationCollator()

    use_weighted_sampler = bool(cfg.get("use_weighted_sampler", False))
    sampler_power = float(cfg.get("sampler_power", 1.0))

    sampler = None
    if use_weighted_sampler:
        train_labels = train_ds.df[cfg["label_col"]].astype(int).tolist()
        sampler, sampler_class_weights = build_weighted_sampler(
            train_labels,
            num_classes=int(cfg["num_classes"]),
            power=sampler_power,
        )
        save_json(
            {
                "sampler_class_weights": sampler_class_weights.tolist(),
                "sampler_power": sampler_power,
            },
            output_dir / "sampler_info.json",
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    param_info = count_trainable_parameters(model)
    save_json(param_info, output_dir / "parameter_report.json")

    if hasattr(model, "replaced_lora_modules"):
        save_json({"replaced_lora_modules": model.replaced_lora_modules}, output_dir / "lora_modules.json")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check config.")

    optimizer = AdamW(
        trainable_params,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    total_train_steps = math.ceil(len(train_loader) / int(cfg["grad_accum_steps"])) * int(cfg["epochs"])
    warmup_steps = int(total_train_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    use_fp16 = (cfg["mixed_precision"] == "fp16") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    class_weights = None
    if str(cfg.get("class_weight_mode", "")).lower() == "balanced":
        y = train_ds.df[cfg["label_col"]].astype(int).tolist()
        classes = np.arange(int(cfg["num_classes"]))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    loss_fn = build_loss(
        loss_type=str(cfg.get("loss_type", "ce")),
        class_weights=class_weights,
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
    )

    best_metric_name = str(cfg.get("metric_for_best", "balanced_accuracy"))
    early_stopper = EarlyStopper(
        patience=int(cfg.get("early_stop_patience", 6)),
        mode="max",
    )

    history = []
    best_value = -1e9

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}"), start=1):
            labels = torch.tensor(batch.labels, dtype=torch.long, device=device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                outputs = model(
                    images=batch.images,
                    image_counts=batch.image_counts,
                    metrics=batch.metrics,
                    aux_images=batch.aux_images,
                    aux_image_counts=batch.aux_image_counts,
                )
                logits = outputs["logits"]

            loss = loss_fn(logits.float(), labels)
            loss = loss / int(cfg["grad_accum_steps"])

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item()) * labels.size(0) * int(cfg["grad_accum_steps"])
            seen += labels.size(0)

            if step % int(cfg["grad_accum_steps"]) == 0 or step == len(train_loader):
                if use_fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg["max_grad_norm"]))

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_loss = running_loss / max(seen, 1)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            label_names=label_names,
            use_fp16=use_fp16,
            loss_fn=loss_fn,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, (dict, list))}
        }
        history.append(record)
        save_json({"history": history}, output_dir / "history.json")

        ckpt = build_checkpoint(
            model=model,
            cfg=cfg,
            label_names=label_names,
            epoch=epoch,
            best_metric_name=best_metric_name,
            param_info=param_info,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
        )

        torch.save(ckpt, output_dir / "last.pt")
        if bool(cfg.get("save_every_epoch", False)):
            torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pt")

        current_value = float(val_metrics[best_metric_name])
        if current_value > best_value:
            best_value = current_value
            torch.save(ckpt, output_dir / "best.pt")
            save_json(val_metrics, output_dir / "best_val_metrics.json")

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        should_stop = early_stopper.step(current_value)
        if should_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best {best_metric_name}={best_value:.4f}")


if __name__ == "__main__":
    main()