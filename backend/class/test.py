from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ClassificationCollator, OCTA500ClassificationDataset
from src.lora import load_lora_state_dict
from src.metrics import compute_metrics
from src.model import build_model
from src.utils import load_yaml, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def load_checkpoint_into_model(model, ckpt: dict):
    save_type = ckpt.get("save_type", "")
    model_type = ckpt.get("model_type", "")

    if save_type == "full_model" or model_type in {"timm", "dual_timm"}:
        if "model_state_dict" not in ckpt:
            raise RuntimeError("Checkpoint marked as full_model but missing model_state_dict.")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return

    if save_type in {"head_only", "head_plus_lora"} or "head_state_dict" in ckpt:
        head_state = ckpt["head_state_dict"]

        model.norm.load_state_dict(
            {
                "weight": head_state["norm.weight"],
                "bias": head_state["norm.bias"],
            }
        )
        model.classifier.load_state_dict(
            {
                "weight": head_state["classifier.weight"],
                "bias": head_state["classifier.bias"],
            }
        )

        if "lora_state_dict" in ckpt and ckpt["lora_state_dict"]:
            load_lora_state_dict(model, ckpt["lora_state_dict"])
        return

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return

    raise RuntimeError(
        f"Unsupported checkpoint format. Keys={list(ckpt.keys())}, "
        f"save_type={save_type}, model_type={model_type}"
    )


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    ckpt_path = args.checkpoint or str(Path(cfg["output_dir"]) / "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_names = ckpt["label_names"]

    sample_mode = str(cfg.get("sample_mode", "image"))
    metrics_cols = cfg.get("metrics_cols", []) or []
    metrics_mean = np.array(ckpt.get("metrics_mean"), dtype=np.float32) if ckpt.get("metrics_mean") is not None else None
    metrics_std = np.array(ckpt.get("metrics_std"), dtype=np.float32) if ckpt.get("metrics_std") is not None else None

    split_to_csv = {
        "train": cfg["train_csv"],
        "val": cfg["val_csv"],
        "test": cfg["test_csv"],
    }

    ds = OCTA500ClassificationDataset(
        split_to_csv[args.split],
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

    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=ClassificationCollator(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint_into_model(model, ckpt)
    model.eval()

    rows = []
    y_true, y_pred = [], []

    use_fp16 = torch.cuda.is_available()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"test:{args.split}"):
            labels = torch.tensor(batch.labels, dtype=torch.long, device=device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(
                    images=batch.images,
                    image_counts=batch.image_counts,
                    metrics=batch.metrics,
                    aux_images=batch.aux_images,
                    aux_image_counts=batch.aux_image_counts,
                )["logits"]

            probs = torch.softmax(logits.float(), dim=-1)
            preds = probs.argmax(dim=-1)

            for i, sid in enumerate(batch.sample_ids):
                row = {
                    "sample_id": sid,
                    "true_id": int(labels[i].item()),
                    "true_name": label_names[int(labels[i].item())],
                    "pred_id": int(preds[i].item()),
                    "pred_name": label_names[int(preds[i].item())],
                }
                for j, name in enumerate(label_names):
                    row[f"prob_{name}"] = float(probs[i, j].item())
                rows.append(row)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, label_names)

    out_dir = Path(cfg["output_dir"]) / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    save_json(metrics, out_dir / "metrics.json")

    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("Saved:")
    print(out_dir / "predictions.csv")
    print(out_dir / "metrics.json")
    print(out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()