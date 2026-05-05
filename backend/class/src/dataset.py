from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_gray_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img


def build_mask_fusion_image(
    orig_path: str,
    rv_path: str,
    faz_path: str,
    out_size: int = 224,
) -> Image.Image:
    orig = load_gray_image(orig_path).resize((out_size, out_size), Image.BILINEAR)
    rv = load_gray_image(rv_path).resize((out_size, out_size), Image.NEAREST)
    faz = load_gray_image(faz_path).resize((out_size, out_size), Image.NEAREST)

    orig_np = np.array(orig, dtype=np.uint8)
    rv_np = np.array(rv, dtype=np.uint8)
    faz_np = np.array(faz, dtype=np.uint8)

    rv_bin = np.where(rv_np > 0, 255, 0).astype(np.uint8)
    faz_bin = np.where(faz_np > 0, 255, 0).astype(np.uint8)

    rgb = np.stack([orig_np, rv_bin, faz_bin], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def build_pure_mask_image(
    rv_path: str,
    faz_path: str,
    out_size: int = 224,
) -> Image.Image:
    rv = load_gray_image(rv_path).resize((out_size, out_size), Image.NEAREST)
    faz = load_gray_image(faz_path).resize((out_size, out_size), Image.NEAREST)

    rv_np = np.array(rv, dtype=np.uint8)
    faz_np = np.array(faz, dtype=np.uint8)

    rv_bin = np.where(rv_np > 0, 255, 0).astype(np.uint8)
    faz_bin = np.where(faz_np > 0, 255, 0).astype(np.uint8)
    union_bin = np.maximum(rv_bin, faz_bin).astype(np.uint8)

    rgb = np.stack([rv_bin, faz_bin, union_bin], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


class OCTA500ClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_cols: List[str],
        label_col: str,
        label_name_col: str,
        sample_mode: str = "image",
        metrics_cols: Optional[List[str]] = None,
        mask_size: int = 224,
        metrics_mean: Optional[np.ndarray] = None,
        metrics_std: Optional[np.ndarray] = None,
        aux_mode: str = "fusion",
    ):
        self.df = pd.read_csv(csv_path)
        self.image_cols = list(image_cols)
        self.label_col = label_col
        self.label_name_col = label_name_col
        self.sample_mode = sample_mode
        self.metrics_cols = metrics_cols or []
        self.mask_size = int(mask_size)
        self.aux_mode = str(aux_mode)

        self.label_names = [
            x
            for _, x in sorted(
                self.df[[self.label_col, self.label_name_col]]
                .drop_duplicates()
                .values.tolist()
            )
        ]

        self.metrics_mean = metrics_mean
        self.metrics_std = metrics_std

    def __len__(self):
        return len(self.df)

    def _load_main_images(self, row) -> List[Image.Image]:
        imgs = []
        for c in self.image_cols:
            imgs.append(load_rgb_image(row[c]))
        return imgs

    def _load_aux_images(self, row) -> Optional[List[Image.Image]]:
        if self.sample_mode == "image_mask_metrics":
            if self.aux_mode == "fusion":
                aux = build_mask_fusion_image(
                    orig_path=row["orig_path"],
                    rv_path=row["rv_path"],
                    faz_path=row["faz_path"],
                    out_size=self.mask_size,
                )
            elif self.aux_mode == "pure_mask":
                aux = build_pure_mask_image(
                    rv_path=row["rv_path"],
                    faz_path=row["faz_path"],
                    out_size=self.mask_size,
                )
            else:
                raise ValueError(f"Unsupported aux_mode: {self.aux_mode}")
            return [aux]
        return None

    def _load_metrics(self, row):
        if not self.metrics_cols:
            return None
        x = row[self.metrics_cols].astype(float).to_numpy(dtype=np.float32)
        if self.metrics_mean is not None and self.metrics_std is not None:
            x = (x - self.metrics_mean) / self.metrics_std
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        sample_id = int(row["sample_id"])
        label = int(row[self.label_col])
        label_name = str(row[self.label_name_col])

        images = self._load_main_images(row)
        aux_images = self._load_aux_images(row)
        metrics = self._load_metrics(row)

        return {
            "sample_id": sample_id,
            "images": images,
            "aux_images": aux_images,
            "label": label,
            "label_name": label_name,
            "metrics": metrics,
        }


@dataclass
class ClassificationBatch:
    sample_ids: List[int]
    images: List[Image.Image]
    image_counts: List[int]
    aux_images: Optional[List[Image.Image]]
    aux_image_counts: Optional[List[int]]
    labels: List[int]
    metrics: Optional[torch.Tensor]


class ClassificationCollator:
    def __call__(self, batch) -> ClassificationBatch:
        sample_ids = []
        images = []
        image_counts = []
        aux_images = []
        aux_image_counts = []
        labels = []
        metrics = []

        has_aux = batch[0]["aux_images"] is not None
        has_metrics = batch[0]["metrics"] is not None

        for item in batch:
            sample_ids.append(item["sample_id"])
            image_counts.append(len(item["images"]))
            images.extend(item["images"])
            labels.append(item["label"])

            if has_aux:
                aux_image_counts.append(len(item["aux_images"]))
                aux_images.extend(item["aux_images"])

            if has_metrics:
                metrics.append(item["metrics"])

        metrics_tensor = None
        if has_metrics:
            metrics_tensor = torch.tensor(np.stack(metrics, axis=0), dtype=torch.float32)

        return ClassificationBatch(
            sample_ids=sample_ids,
            images=images,
            image_counts=image_counts,
            aux_images=aux_images if has_aux else None,
            aux_image_counts=aux_image_counts if has_aux else None,
            labels=labels,
            metrics=metrics_tensor,
        )