# -*- coding: utf-8 -*-
import os
import json
import argparse
import contextlib
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from torch.amp import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from dataset import OCTADataset, load_labels_multi

try:
    from models import build_model, build_model_multimodal_text
except Exception:
    from models import build_model
    build_model_multimodal_text = None


# ---------------- 中文映射（疾病/指标） ----------------
DISEASE_CN = {
    "AMD": "老年性黄斑变性",
    "CNV": "脉络膜新生血管",
    "CSC": "中央浆液性脉络膜视网膜病变",
    "DR": "糖尿病视网膜病变",
    "NORMAL": "正常",
    "OTHERS": "其他",
    "RVO": "视网膜静脉阻塞",
}

METRIC_CN = {
    "faz_area_px": "FAZ面积(px)",
    "faz_perim_px": "FAZ周长(px)",
    "faz_circularity": "FAZ圆形度",
    "faz300_sim_density": "FAZ周边密度(300px)",
    "rv_density": "血管密度",
    "rv_flow_area_px": "血管流域面积(px)",
    "rv_line_density_px-1": "血管线密度(px⁻¹)",
    "rv_branch_points": "血管分支点数",
}

ORDERED_METRIC_KEYS = [
    "faz_area_px",
    "faz_perim_px",
    "faz_circularity",
    "faz300_sim_density",
    "rv_density",
    "rv_flow_area_px",
    "rv_line_density_px-1",
    "rv_branch_points",
]


# ---------------- Heatmap / Excel 工具 ----------------
def _guess_id_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    cols = [str(c).strip() for c in df.columns]
    if preferred and preferred in cols:
        return preferred
    for c in ["编号", "image_id", "ID", "Id", "id", "case_id", "CaseID", "Subject", "PatientID"]:
        if c in cols:
            return c
    raise ValueError(f"无法识别 ID 列，当前列名: {list(df.columns)}")


def _to_str_id(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer, float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x).strip()


def _canonicalize_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        if c == "编号":
            continue
        cs = str(c)
        low = cs.lower()

        if cs in METRIC_CN:
            ren[cs] = cs
            continue

        if ("faz" in low and "circular" in low) or ("faz" in low and "几何" in cs):
            ren[cs] = "faz_circularity"; continue
        if ("rv" in low and "密度" in cs) or ("rv" in low and "density" in low and "line" not in low):
            ren[cs] = "rv_density"; continue
        if ("rv" in low and ("branch" in low or "分支" in cs)):
            ren[cs] = "rv_branch_points"; continue
        if ("faz" in low and ("area" in low or "面积" in cs)):
            ren[cs] = "faz_area_px"; continue
        if ("faz" in low and ("perim" in low or "周长" in cs)):
            ren[cs] = "faz_perim_px"; continue
        if ("rv" in low and ("flow" in low or "流域" in cs)):
            ren[cs] = "rv_flow_area_px"; continue
        if ("rv" in low and "line" in low and "density" in low):
            ren[cs] = "rv_line_density_px-1"; continue
        if ("faz" in low and ("300" in low or "周边" in cs) and "density" in low):
            ren[cs] = "faz300_sim_density"; continue

    if ren:
        df = df.rename(columns=ren)

    keep = ["编号"] + [c for c in df.columns if c in METRIC_CN]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def _display_metric_name(key: str) -> str:
    if key in METRIC_CN:
        return METRIC_CN[key]
    rep = key
    rep = rep.replace("faz", "FAZ").replace("rv", "血管")
    rep = rep.replace("area", "面积").replace("perim", "周长")
    rep = rep.replace("circularity", "圆形度").replace("density", "密度")
    rep = rep.replace("branch", "分支").replace("points", "点数").replace("flow", "流域")
    rep = rep.replace("line", "线")
    return rep


def _read_one_metrics_excel(xlsx_path: str, id_col: str = "编号") -> pd.DataFrame:
    if xlsx_path is None or not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"指标文件不存在: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    if len(df) == 0:
        raise ValueError(f"指标文件为空: {xlsx_path}")

    real_id_col = _guess_id_col(df, preferred=id_col)
    df = df.copy()
    df[real_id_col] = df[real_id_col].map(_to_str_id)

    if real_id_col != "编号":
        df = df.rename(columns={real_id_col: "编号"})

    df = _canonicalize_metric_cols(df)

    if "编号" not in df.columns:
        raise ValueError(f"{xlsx_path} 缺少 ID 列")
    if len(df.columns) <= 1:
        raise ValueError(f"{xlsx_path} 未识别到有效指标列")

    df["编号"] = df["编号"].astype(str).str.strip()

    for c in df.columns:
        if c != "编号":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_metrics_from_two_excels(
    metrics_3m_xlsx: str,
    metrics_6m_xlsx: str,
    id_col: str = "编号",
) -> pd.DataFrame:
    df3 = _read_one_metrics_excel(metrics_3m_xlsx, id_col=id_col)
    df6 = _read_one_metrics_excel(metrics_6m_xlsx, id_col=id_col)

    df = pd.concat([df3, df6], ignore_index=True)

    # 同一 ID 若重复，保留最后一个
    df = df.drop_duplicates(subset=["编号"], keep="last").reset_index(drop=True)

    # 确保列顺序：编号 + 8个标准指标
    cols = ["编号"] + [k for k in ORDERED_METRIC_KEYS if k in df.columns]
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]

    return df


def _df_to_tabletext_map(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for _, row in df.iterrows():
        sample_id = str(row["编号"]).strip()
        parts = []
        for k in ORDERED_METRIC_KEYS:
            val = row[k] if k in df.columns else 0.0
            if pd.isna(val):
                val = 0.0
            parts.append(f"{k}: {float(val):.6g}")
        out[sample_id] = "; ".join(parts)
    return out


def load_tabletext_from_two_excels(
    metrics_3m_xlsx: str,
    metrics_6m_xlsx: str,
    id_col: str = "编号",
):
    df3 = _read_one_metrics_excel(metrics_3m_xlsx, id_col=id_col)
    df6 = _read_one_metrics_excel(metrics_6m_xlsx, id_col=id_col)
    text3 = _df_to_tabletext_map(df3)
    text6 = _df_to_tabletext_map(df6)
    return text3, text6


def generate_metric_disease_heatmap(
    out_dir: str,
    id2y: Dict[str, int],
    class_names: List[str],
    metrics_df: Optional[pd.DataFrame] = None,
    metrics_3m_xlsx: Optional[str] = None,
    metrics_6m_xlsx: Optional[str] = None,
    id_col: str = "编号",
):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] 缺少 seaborn/matplotlib，跳过热力图生成。")
        return

    dfm = None
    if metrics_df is not None:
        dfm = metrics_df.copy()
    elif metrics_3m_xlsx is not None and metrics_6m_xlsx is not None:
        try:
            dfm = load_metrics_from_two_excels(metrics_3m_xlsx, metrics_6m_xlsx, id_col=id_col)
        except Exception as e:
            print(f"[WARN] 读取两个 metrics xlsx 失败: {e}")
            dfm = None

    if dfm is None or len(dfm) == 0:
        print("[INFO] 未提供有效的指标表，跳过热力图。")
        return

    real_id_col = _guess_id_col(dfm, preferred=id_col if id_col in dfm.columns else "编号")
    dfm = dfm.copy()
    dfm[real_id_col] = dfm[real_id_col].map(_to_str_id)
    if real_id_col != "编号":
        dfm = dfm.rename(columns={real_id_col: "编号"})
    dfm = _canonicalize_metric_cols(dfm)

    metric_cols = [c for c in dfm.columns if c != "编号"]
    if not metric_cols:
        print("[INFO] 指标列为空，跳过热力图。")
        return

    label_df = pd.DataFrame({"编号": list(id2y.keys()), "y": list(id2y.values())})
    merged = pd.merge(label_df, dfm, on="编号", how="inner")
    if len(merged) < 3:
        print("[INFO] 交集样本过少，跳过热力图。")
        return

    mat = np.zeros((len(metric_cols), len(class_names)), dtype=float)
    for j, cls in enumerate(class_names):
        s = (merged["y"].values == j).astype(float)
        s_ser = pd.Series(s)
        for i, m in enumerate(metric_cols):
            try:
                r = pd.Series(merged[m].astype(float).values).corr(s_ser, method="spearman")
            except Exception:
                r = float("nan")
            mat[i, j] = r if r is not None else float("nan")

    row_labels = [_display_metric_name(m) for m in metric_cols]
    col_labels = [f"{en}（{DISEASE_CN.get(en, en)}）" for en in class_names]
    corr_df = pd.DataFrame(mat, index=row_labels, columns=col_labels)

    csv_path = os.path.join(out_dir, "metric_disease_corr_spearman.csv")
    corr_df.to_csv(csv_path, encoding="utf-8-sig")

    h, w = max(4, 0.5 * len(row_labels)), max(4, 0.8 * len(col_labels))
    plt.figure(figsize=(w, h))
    ax = sns.heatmap(
        corr_df, cmap="coolwarm", vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        cbar_kws={"label": "Spearman 相关系数"}
    )
    ax.set_xlabel("病变（中文）")
    ax.set_ylabel("报告指标（中文）")
    plt.tight_layout()

    png_path = os.path.join(out_dir, "metric_disease_heatmap_spearman.png")
    pdf_path = os.path.join(out_dir, "metric_disease_heatmap_spearman.pdf")
    plt.savefig(png_path, dpi=300)
    try:
        plt.savefig(pdf_path)
    except Exception:
        pass
    plt.close()
    print(f"[OK] 已保存指标-病变热力图：{png_path}")


# ---------------- Utils ----------------
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def autocast_ctx():
    if torch.cuda.is_available():
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except TypeError:
            return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _make_outdir(base_out: str, root_path: str, arch: str, mode: str) -> str:
    tag = os.path.basename(os.path.abspath(root_path))
    out_dir = os.path.join(base_out, f"{tag}_{arch}_{mode}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _eval_from_logits(y_true: np.ndarray, logits: np.ndarray, classes: List[str]) -> Dict:
    proba = _softmax(logits.astype(np.float64))
    y_pred = proba.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    uni = np.unique(y_true)
    if uni.size < 2:
        roc_auc_macro = float("nan")
        ap_macro = float("nan")
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(classes))))
        roc_auc_macro = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        ap_macro = average_precision_score(y_bin, proba, average="macro")

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro_ovr": float(roc_auc_macro),
        "ap_macro_ovr": float(ap_macro),
        "confusion_matrix": cm.tolist(),
        "proba": proba,
        "y_pred": y_pred,
    }


@torch.no_grad()
def evaluate(model, loader, device, mode, num_classes) -> Dict:
    model.eval()
    ys_all, logits_all = [], []

    for batch in loader:
        if mode == "img":
            x, y, _ids, _fov = batch
            x = x.to(device)
            y = y.to(device)
            with autocast_ctx():
                logits = model(x)

        elif mode == "img+metrics":
            x, y, _ids, _fov, m = batch
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            with autocast_ctx():
                logits = model(x, m)

        else:
            x, y, _ids, _fov, texts = batch
            x = x.to(device)
            y = y.to(device)
            with autocast_ctx():
                logits = model(x, texts)

        ys_all.append(y.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())

    ys = np.concatenate(ys_all)
    logits = np.concatenate(logits_all)
    return _eval_from_logits(ys, logits, classes=list(range(num_classes)))


def _collate(mode: str):
    def _fn(batch):
        imgs, labels, ids, fovs, extras = [], [], [], [], []
        for item in batch:
            if len(item) == 5:
                img, lab, id_, fov_, extra = item
            elif len(item) == 4:
                img, lab, id_, fov_ = item
                extra = 0.0 if mode == "img+metrics" else "no_features"
            elif len(item) == 3:
                img, lab, id_ = item
                fov_ = torch.tensor(0, dtype=torch.long)
                extra = 0.0 if mode == "img+metrics" else "no_features"
            else:
                raise ValueError(f"Unexpected item length: {len(item)}")

            imgs.append(img)
            labels.append(lab)
            ids.append(id_)
            fovs.append(fov_)
            extras.append(extra)

        if mode == "img":
            return torch.stack(imgs), torch.stack(labels), ids, torch.stack(fovs)
        else:
            if mode == "img+metrics":
                extras = torch.stack([
                    e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)
                    for e in extras
                ])
            return torch.stack(imgs), torch.stack(labels), ids, torch.stack(fovs), extras

    return _fn


# ---------------- CV core ----------------
def kfold_train_one_root(
    data_root: str,
    labels_xlsx: str,
    arch: str,
    mode: str,
    out_base: str,
    label_col: str = "Disease",
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    metrics_3m_xlsx: Optional[str] = None,
    metrics_6m_xlsx: Optional[str] = None,
    id_col: str = "编号",
    cv_folds: int = 5,
    cv_seed: int = 42,
):
    assert mode in ("img", "img+metrics", "img+tabletext")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = _make_outdir(out_base, data_root, arch, mode)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = os.path.basename(os.path.abspath(data_root))

    id2y_full, le, class_names_all = load_labels_multi(labels_xlsx, label_col=label_col)

    metrics_df = None
    text3 = text6 = None

    if mode == "img+metrics":
        if metrics_3m_xlsx is None or metrics_6m_xlsx is None:
            raise ValueError("img+metrics 模式需要同时提供 --metrics_3m 和 --metrics_6m")
        metrics_df = load_metrics_from_two_excels(metrics_3m_xlsx, metrics_6m_xlsx, id_col=id_col)
        print(f"[INFO] 已载入数值指标表: {metrics_df.shape}")

    if mode == "img+tabletext":
        if metrics_3m_xlsx is None or metrics_6m_xlsx is None:
            raise ValueError("img+tabletext 模式需要同时提供 --metrics_3m 和 --metrics_6m")
        text3, text6 = load_tabletext_from_two_excels(metrics_3m_xlsx, metrics_6m_xlsx, id_col=id_col)
        print(f"[INFO] 已载入 tabletext: 3M={len(text3)} 条, 6M={len(text6)} 条")

    probe_train = OCTADataset(
        [data_root], "train", id2y_full, mode,
        metrics_df=metrics_df, metrics_norm=None,
        tabletext_3m=text3, tabletext_6m=text6
    )
    if len(probe_train) == 0:
        raise RuntimeError(f"[{tag}] 训练集为空，请检查 {data_root}/train 是否包含 FAZ/RV 成对图像")

    y_old_all = np.array([probe_train.items[i][3] for i in range(len(probe_train))], dtype=np.int64)
    cnt_old = Counter(y_old_all.tolist())
    present_old_idx = sorted(cnt_old.keys())
    class_names = [class_names_all[i] for i in present_old_idx]

    print(f"[DEBUG:{tag}] train class distribution (old index -> count): {dict(cnt_old)}")
    print(f"[DEBUG:{tag}] present classes -> {class_names}")

    old2new = {old: new for new, old in enumerate(present_old_idx)}
    id2y = {k: old2new[v] for k, v in id2y_full.items() if v in old2new}
    num_classes = len(class_names)

    base_train_ds = OCTADataset(
        [data_root], "train", id2y, mode,
        metrics_df=metrics_df, metrics_norm=None,
        tabletext_3m=text3, tabletext_6m=text6
    )
    y_all = np.array([base_train_ds.items[i][3] for i in range(len(base_train_ds))], dtype=np.int64)
    n_train = len(base_train_ds)
    print(f"[DATA:{tag}] train={n_train}  classes={num_classes} -> {class_names}")

    min_count = min(Counter(y_all).values())
    if cv_folds > min_count:
        print(f"[WARN] Reducing cv_folds from {cv_folds} to {min_count} because min class has only {min_count} samples.")
        cv_folds = min_count

    oof_logits = np.full((n_train, num_classes), np.nan, dtype=np.float32)
    oof_true = y_all.copy()

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)

    test_preds_accum = []
    fold_summaries = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_all), start=1):
        print(f"\n========== CV Fold {fold_id}/{cv_folds} ({tag}) ==========")

        metrics_norm = None
        if mode == "img+metrics":
            tmp_ds = OCTADataset(
                [data_root], "train", id2y, mode,
                metrics_df=metrics_df, metrics_norm=None,
                tabletext_3m=text3, tabletext_6m=text6
            )
            if hasattr(tmp_ds, "metrics") and tmp_ds.metrics is not None and len(tr_idx) > 0:
                arr = np.stack([tmp_ds.metrics[i] for i in tr_idx], axis=0)
                mu = arr.mean(axis=0)
                sigma = arr.std(axis=0)
                metrics_norm = {
                    c: (float(mu[i]), float(sigma[i]) + 1e-8)
                    for i, c in enumerate(tmp_ds.metrics_cols)
                }

        fold_train_full = OCTADataset(
            [data_root], "train", id2y, mode,
            metrics_df=metrics_df, metrics_norm=metrics_norm,
            tabletext_3m=text3, tabletext_6m=text6
        )
        ds_tr = Subset(fold_train_full, tr_idx)
        ds_va = Subset(fold_train_full, va_idx)
        ds_te = OCTADataset(
            [data_root], "test", id2y, mode,
            metrics_df=metrics_df, metrics_norm=metrics_norm,
            tabletext_3m=text3, tabletext_6m=text6
        )

        collate = _collate(mode)
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate)
        loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
        loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)

        if mode == "img+tabletext" and build_model_multimodal_text is not None:
            model = build_model_multimodal_text(arch, num_classes=len(class_names)).to(device)
        else:
            metrics_dim = 0
            if mode == "img+metrics":
                if metrics_df is None:
                    raise RuntimeError("img+metrics 模式下 metrics_df 不应为空")
                metrics_dim = len([c for c in metrics_df.columns if c != "编号"])
            model = build_model(
                arch,
                num_classes=len(class_names),
                use_metrics=(mode == "img+metrics"),
                metrics_dim=metrics_dim
            ).to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=torch.cuda.is_available())

        best_val_acc = -1.0
        best_val_payload = None
        best_ckpt = os.path.join(out_dir, f"fold{fold_id}_best.pt")

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0

            pbar = tqdm(loader_tr, desc=f"[{tag}][Fold{fold_id}][Epoch {epoch:03d}/{epochs}]", leave=False)

            for batch in pbar:
                optim.zero_grad(set_to_none=True)

                if mode == "img":
                    x, y, _ids, _fov = batch
                    x = x.to(device)
                    y = y.to(device)
                    with autocast_ctx():
                        logits = model(x)
                        loss = criterion(logits, y)

                elif mode == "img+metrics":
                    x, y, _ids, _fov, m = batch
                    x = x.to(device)
                    y = y.to(device)
                    m = m.to(device)
                    with autocast_ctx():
                        logits = model(x, m)
                        loss = criterion(logits, y)

                else:
                    x, y, _ids, _fov, texts = batch
                    x = x.to(device)
                    y = y.to(device)
                    with autocast_ctx():
                        logits = model(x, texts)
                        loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                loss_val = loss.item()
                running += loss_val * (x.size(0) if hasattr(x, "size") else 1)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            pbar.close()
            sched.step()
            tr_loss = running / max(1, len(loader_tr.dataset))

            val_res = evaluate(model, loader_va, device, mode, num_classes)
            print(
                f"[{tag}][Fold{fold_id}][{epoch:03d}] loss={tr_loss:.4f}  "
                f"val_acc={val_res['acc']:.4f}  val_macroF1={val_res['macro_f1']:.4f}"
            )

            if val_res["acc"] > best_val_acc:
                best_val_acc = val_res["acc"]
                test_res = evaluate(model, loader_te, device, mode, num_classes)
                payload = {
                    "arch": arch,
                    "mode": mode,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_res,
                    "test_metrics": test_res,
                }
                torch.save(payload, best_ckpt)
                best_val_payload = payload
                print(f"  ↳ [BEST@{epoch}] test_acc={test_res['acc']:.4f}  test_macroF1={test_res['macro_f1']:.4f}")

        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])

        model.eval()
        ys_fold, logits_fold = [], []
        for batch in loader_va:
            with torch.no_grad():
                if mode == "img":
                    x, y, _ids, _fov = batch
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx():
                        lo = model(x)

                elif mode == "img+metrics":
                    x, y, _ids, _fov, m = batch
                    x, y, m = x.to(device), y.to(device), m.to(device)
                    with autocast_ctx():
                        lo = model(x, m)

                else:
                    x, y, _ids, _fov, texts = batch
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx():
                        lo = model(x, texts)

                ys_fold.append(y.detach().cpu().numpy())
                logits_fold.append(lo.float().detach().cpu().numpy())

        ys_fold = np.concatenate(ys_fold)
        logits_fold = np.concatenate(logits_fold)
        oof_logits[va_idx] = logits_fold

        if best_val_payload is None:
            val_now = evaluate(model, loader_va, device, mode, num_classes)
            test_now = evaluate(model, loader_te, device, mode, num_classes)
            best_val_payload = {"val_metrics": val_now, "test_metrics": test_now}

        fold_summaries.append({
            "fold": fold_id,
            "val_acc": float(best_val_payload["val_metrics"].get("acc", float("nan"))),
            "val_macroF1": float(best_val_payload["val_metrics"].get("macro_f1", float("nan"))),
            "test_acc": float(best_val_payload["test_metrics"].get("acc", float("nan"))),
            "test_macroF1": float(best_val_payload["test_metrics"].get("macro_f1", float("nan"))),
        })

        ys_te_all, logits_te_all = [], []
        for batch in loader_te:
            with torch.no_grad():
                if mode == "img":
                    x, y, _ids, _fov = batch
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx():
                        lo = model(x)

                elif mode == "img+metrics":
                    x, y, _ids, _fov, m = batch
                    x, y, m = x.to(device), y.to(device), m.to(device)
                    with autocast_ctx():
                        lo = model(x, m)

                else:
                    x, y, _ids, _fov, texts = batch
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx():
                        lo = model(x, texts)

                ys_te_all.append(y.detach().cpu().numpy())
                logits_te_all.append(lo.float().detach().cpu().numpy())

        logits_te = np.concatenate(logits_te_all)
        test_preds_accum.append(_softmax(logits_te.astype(np.float64)))

    oof_proba = _softmax(oof_logits.astype(np.float64))
    y_pred_enc = oof_proba.argmax(axis=1)

    acc = accuracy_score(oof_true, y_pred_enc)
    bacc = balanced_accuracy_score(oof_true, y_pred_enc)
    f1_m = f1_score(oof_true, y_pred_enc, average="macro")
    f1_w = f1_score(oof_true, y_pred_enc, average="weighted")

    try:
        y_bin = label_binarize(oof_true, classes=list(range(num_classes)))
        roc_auc_macro = roc_auc_score(y_bin, oof_proba, average="macro", multi_class="ovr")
        ap_macro = average_precision_score(y_bin, oof_proba, average="macro")
    except Exception:
        roc_auc_macro, ap_macro = float("nan"), float("nan")

    cm = confusion_matrix(oof_true, y_pred_enc, labels=list(range(num_classes))).tolist()

    class_names = list(class_names)
    oof_df = pd.DataFrame({f"proba_{c}": oof_proba[:, i] for i, c in enumerate(class_names)})
    oof_df["y_true"] = [class_names[i] for i in oof_true]
    oof_df["y_pred"] = [class_names[i] for i in y_pred_enc]

    report_txt = classification_report(
        [class_names[i] for i in oof_true],
        [class_names[i] for i in y_pred_enc],
        labels=class_names,
        target_names=class_names
    )

    Path(os.path.join(out_dir, f"oof_predictions_{tag}.csv")).write_text(
        oof_df.to_csv(index=False), encoding="utf-8"
    )
    Path(os.path.join(out_dir, f"classification_report_{tag}.txt")).write_text(
        report_txt, encoding="utf-8"
    )

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "roc_auc_macro_ovr": float(roc_auc_macro),
        "ap_macro_ovr": float(ap_macro),
        "n_samples": int(len(oof_true)),
        "n_classes": int(num_classes),
        "classes": class_names,
        "confusion_matrix": cm
    }
    Path(os.path.join(out_dir, f"metrics_{tag}.json")).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    fold_df = pd.DataFrame(fold_summaries)
    if len(fold_df) > 0:
        fold_df.loc["mean"] = fold_df.mean(numeric_only=True)
    fold_df.to_csv(os.path.join(out_dir, f"cv_{len(fold_summaries)}folds_summary.csv"), index=True)

    print("\n== OOF (Stratified K-Fold) ==")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n折概要：\n", fold_df)

    if len(test_preds_accum) > 0:
        te_avg = np.mean(test_preds_accum, axis=0)

        ds_te = OCTADataset(
            [data_root], "test", id2y, mode,
            metrics_df=metrics_df, metrics_norm=None,
            tabletext_3m=text3, tabletext_6m=text6
        )
        loader_te = DataLoader(
            ds_te, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=_collate(mode)
        )

        ys_te_all = []
        for batch in loader_te:
            y = batch[1]
            ys_te_all.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

        y_true_test = np.concatenate(ys_te_all)
        y_pred_test = te_avg.argmax(axis=1)
        test_metrics = {
            "acc": float(accuracy_score(y_true_test, y_pred_test)),
            "macro_f1": float(f1_score(y_true_test, y_pred_test, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_true_test, y_pred_test)),
            "cm": confusion_matrix(y_true_test, y_pred_test, labels=list(range(num_classes))).tolist()
        }
        Path(os.path.join(out_dir, f"metrics_test_ensemble_{tag}.json")).write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n== TEST (fold-avg ensemble) ==")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    try:
        generate_metric_disease_heatmap(
            out_dir=out_dir,
            id2y=id2y,
            class_names=class_names,
            metrics_df=(metrics_df if mode == "img+metrics" else None),
            metrics_3m_xlsx=metrics_3m_xlsx,
            metrics_6m_xlsx=metrics_6m_xlsx,
            id_col=id_col,
        )
    except Exception as e:
        print(f"[WARN] 生成指标-病变热力图失败：{e}")


# ---------------- Main ----------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_roots", nargs="+", required=True, help="一个或多个数据根目录（每个都需含 train/test 子目录）")
    ap.add_argument("--labels", type=str, required=True, help="标签 Excel（含 ID 与 Disease 列/或指定 label_col）")
    ap.add_argument("--label_col", type=str, default="Disease")

    ap.add_argument("--mode", type=str, default="img", choices=["img", "img+metrics", "img+tabletext"])

    ap.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_b_16", "convnext_tiny", "swin_t", "rvmamba"]
    )

    ap.add_argument("--out", type=str, default="outputs_mmtext_cv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # 新版：直接接收两个 xlsx
    ap.add_argument("--metrics_3m", type=str, default=None, help="3M 指标文件，如 output/3m/metrics_3m.xlsx")
    ap.add_argument("--metrics_6m", type=str, default=None, help="6M 指标文件，如 output/6m/metrics_6m.xlsx")
    ap.add_argument("--id_col", type=str, default="image_id", help="两个 xlsx 中的 ID 列名，默认 image_id")

    ap.add_argument("--cv_folds", type=int, default=5, help=">1 启用 Stratified K 折；若最小类样本不足会自动下调")
    ap.add_argument("--cv_seed", type=int, default=42)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for root in args.data_roots:
        kfold_train_one_root(
            data_root=root,
            labels_xlsx=args.labels,
            arch=args.arch,
            mode=args.mode,
            out_base=args.out,
            label_col=args.label_col,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            metrics_3m_xlsx=args.metrics_3m,
            metrics_6m_xlsx=args.metrics_6m,
            id_col=args.id_col,
            cv_folds=max(2, args.cv_folds),
            cv_seed=args.cv_seed,
        )
# python classfy.py \
#   --data_roots /mnt/d/octa_agent/backend/output/3m /mnt/d/octa_agent/backend/output/6m \
#   --labels /mnt/d/octa_agent/backend/Textlabels.xlsx \
#   --mode img+tabletext \
#   --arch resnet50 \
#   --metrics_3m /mnt/d/octa_agent/backend/output/3m/metrics_3m.xlsx \
#   --metrics_6m /mnt/d/octa_agent/backend/output/6m/metrics_6m.xlsx \
#   --id_col image_id \
#   --epochs 30 \
#   --batch_size 16