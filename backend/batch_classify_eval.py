# -*- coding: utf-8 -*-
import os
import json
import base64
import argparse
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


ORDERED_KEYS = [
    "faz_area_px",
    "faz_perim_px",
    "faz_circularity",
    "faz300_sim_density",
    "rv_density",
    "rv_flow_area_px",
    "rv_line_density_px-1",
    "rv_branch_points",
]


# =========================
# 基础工具
# =========================
def normalize_id(v):
    if pd.isna(v):
        return ""
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
    except Exception:
        pass
    return str(v).strip()


def encode_image_to_dataurl(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + b64


def list_image_files(folder: str):
    if not os.path.isdir(folder):
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        str(p) for p in Path(folder).glob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def build_id_path_map(folder: str):
    out = {}
    for p in list_image_files(folder):
        image_id = Path(p).stem
        out[image_id] = p
    return out


def detect_split_pairs(root_dir: str, split: str):
    faz_dir = os.path.join(root_dir, "faz_label", split)
    rv_dir = os.path.join(root_dir, "rv_label", split)

    faz_map = build_id_path_map(faz_dir)
    rv_map = build_id_path_map(rv_dir)

    common_ids = sorted(set(faz_map.keys()) & set(rv_map.keys()))
    return common_ids, faz_map, rv_map


def detect_flat_pairs(root_dir: str):
    faz_dir = os.path.join(root_dir, "faz_label")
    rv_dir = os.path.join(root_dir, "rv_label")

    faz_map = build_id_path_map(faz_dir)
    rv_map = build_id_path_map(rv_dir)

    common_ids = sorted(set(faz_map.keys()) & set(rv_map.keys()))
    return common_ids, faz_map, rv_map


# =========================
# 读取指标
# =========================
def load_metrics_map(metrics_xlsx: str, id_col: str = None) -> dict:
    df = pd.read_excel(metrics_xlsx)
    df.columns = [str(c).strip() for c in df.columns]

    if id_col is None or id_col not in df.columns:
        for c in ["image_id", "编号", "ID", "id"]:
            if c in df.columns:
                id_col = c
                break

    if id_col is None or id_col not in df.columns:
        raise ValueError(f"指标表中找不到 ID 列，当前列名: {list(df.columns)}")

    df[id_col] = df[id_col].map(normalize_id)

    metrics_map = {}
    for _, row in df.iterrows():
        image_id = row[id_col]
        metrics = {}
        for k in ORDERED_KEYS:
            val = row[k] if k in df.columns else 0.0
            if pd.isna(val):
                val = 0.0
            metrics[k] = float(val)
        metrics_map[image_id] = metrics

    return metrics_map


# =========================
# 读取标签
# =========================
def load_labels_map(labels_xlsx: str, id_col: str = None, label_col: str = None):
    df = pd.read_excel(labels_xlsx)
    df.columns = [str(c).strip() for c in df.columns]

    if id_col is None or id_col not in df.columns:
        for c in ["image_id", "编号", "ID", "id", "case_id", "CaseID"]:
            if c in df.columns:
                id_col = c
                break

    if label_col is None or label_col not in df.columns:
        for c in ["Disease", "label", "Label", "diagnosis", "Diagnosis"]:
            if c in df.columns:
                label_col = c
                break

    if id_col is None or id_col not in df.columns:
        raise ValueError(f"标签表中找不到 ID 列，当前列名: {list(df.columns)}")
    if label_col is None or label_col not in df.columns:
        raise ValueError(f"标签表中找不到标签列，当前列名: {list(df.columns)}")

    df = df.dropna(subset=[id_col, label_col]).copy()
    df[id_col] = df[id_col].map(normalize_id)
    df[label_col] = df[label_col].astype(str).str.strip()

    label_map = {row[id_col]: row[label_col] for _, row in df.iterrows()}
    return label_map, id_col, label_col


# =========================
# 调分类接口
# =========================
def call_classify_api(api_url: str, scan_type: str, rv_path: str, faz_path: str, metrics: dict, timeout=120):
    payload = {
        "scan_type": scan_type,
        "rv_mask_base64": encode_image_to_dataurl(rv_path),
        "faz_mask_base64": encode_image_to_dataurl(faz_path),
        "metrics": metrics,
    }

    resp = requests.post(api_url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"API failed: {json.dumps(data, ensure_ascii=False)}")
    return data


# =========================
# 单目录批量分类
# =========================
def classify_one_group(
    split_name: str,
    ids: list,
    faz_map: dict,
    rv_map: dict,
    metrics_map: dict,
    label_map: dict,
    api_url: str,
    scan_type: str,
):
    rows = []
    failed = []

    total = len(ids)
    for idx, image_id in enumerate(ids, 1):
        try:
            if image_id not in metrics_map:
                raise RuntimeError(f"metrics 中缺少 image_id={image_id}")

            result = call_classify_api(
                api_url=api_url,
                scan_type=scan_type,
                rv_path=rv_map[image_id],
                faz_path=faz_map[image_id],
                metrics=metrics_map[image_id],
            )

            pred = result.get("prediction", {})
            meta = result.get("meta", {})

            true_label = label_map.get(image_id, None)
            pred_label = pred.get("label_en", "")

            row = {
                "scan_type": scan_type,
                "split": split_name,
                "image_id": image_id,
                "y_true": true_label,
                "y_pred": pred_label,
                "correct": int(true_label == pred_label) if true_label is not None else None,
                "label_cn": pred.get("label_cn", ""),
                "confidence": pred.get("confidence", None),
                "ensemble_count": meta.get("ensemble_count", None),
                "mode_used": meta.get("mode_used", ""),
            }

            distribution = pred.get("distribution", {})
            for cls_name, prob in distribution.items():
                row[f"prob_{cls_name}"] = prob

            rows.append(row)
            print(f"[{scan_type}][{idx}/{total}] OK   {split_name}/{image_id} -> pred={pred_label}, true={true_label}")

        except Exception as e:
            failed.append({
                "scan_type": scan_type,
                "split": split_name,
                "image_id": image_id,
                "rv_path": rv_map.get(image_id, ""),
                "faz_path": faz_map.get(image_id, ""),
                "error": str(e),
            })
            print(f"[{scan_type}][{idx}/{total}] FAIL {split_name}/{image_id} -> {e}")

    return rows, failed


# =========================
# 评估
# =========================
def evaluate_predictions(df: pd.DataFrame):
    eval_rows = []

    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    if len(valid) == 0:
        return pd.DataFrame(columns=["scope", "scan_type", "split", "n", "acc", "macro_f1"])

    # overall
    acc = accuracy_score(valid["y_true"], valid["y_pred"])
    macro_f1 = f1_score(valid["y_true"], valid["y_pred"], average="macro")
    eval_rows.append({
        "scope": "overall",
        "scan_type": "ALL",
        "split": "ALL",
        "n": len(valid),
        "acc": acc,
        "macro_f1": macro_f1,
    })

    # per scan_type
    for scan_type, sub in valid.groupby("scan_type"):
        acc = accuracy_score(sub["y_true"], sub["y_pred"])
        macro_f1 = f1_score(sub["y_true"], sub["y_pred"], average="macro")
        eval_rows.append({
            "scope": "by_scan_type",
            "scan_type": scan_type,
            "split": "ALL",
            "n": len(sub),
            "acc": acc,
            "macro_f1": macro_f1,
        })

    # per split
    for split, sub in valid.groupby("split"):
        acc = accuracy_score(sub["y_true"], sub["y_pred"])
        macro_f1 = f1_score(sub["y_true"], sub["y_pred"], average="macro")
        eval_rows.append({
            "scope": "by_split",
            "scan_type": "ALL",
            "split": split,
            "n": len(sub),
            "acc": acc,
            "macro_f1": macro_f1,
        })

    # per scan_type + split
    for (scan_type, split), sub in valid.groupby(["scan_type", "split"]):
        acc = accuracy_score(sub["y_true"], sub["y_pred"])
        macro_f1 = f1_score(sub["y_true"], sub["y_pred"], average="macro")
        eval_rows.append({
            "scope": "by_scan_type_and_split",
            "scan_type": scan_type,
            "split": split,
            "n": len(sub),
            "acc": acc,
            "macro_f1": macro_f1,
        })

    return pd.DataFrame(eval_rows)


def save_confusion_matrices(df: pd.DataFrame, save_dir: str):
    valid = df.dropna(subset=["y_true", "y_pred"]).copy()
    if len(valid) == 0:
        return

    labels = sorted(set(valid["y_true"].tolist()) | set(valid["y_pred"].tolist()))
    cm = confusion_matrix(valid["y_true"], valid["y_pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    cm_df.to_excel(os.path.join(save_dir, "confusion_matrix_overall.xlsx"))

    for scan_type, sub in valid.groupby("scan_type"):
        labels_sub = sorted(set(sub["y_true"].tolist()) | set(sub["y_pred"].tolist()))
        cm_sub = confusion_matrix(sub["y_true"], sub["y_pred"], labels=labels_sub)
        cm_sub_df = pd.DataFrame(
            cm_sub,
            index=[f"true_{x}" for x in labels_sub],
            columns=[f"pred_{x}" for x in labels_sub]
        )
        cm_sub_df.to_excel(os.path.join(save_dir, f"confusion_matrix_{scan_type.lower()}.xlsx"))


# =========================
# 处理单个 scan root
# =========================
def process_one_scan_root(root_dir: str, scan_type: str, api_url: str, metrics_xlsx: str, label_map: dict):
    metrics_map = load_metrics_map(metrics_xlsx, id_col=None)

    all_rows = []
    all_failed = []

    has_split_dirs = (
        os.path.isdir(os.path.join(root_dir, "faz_label", "train")) or
        os.path.isdir(os.path.join(root_dir, "faz_label", "test")) or
        os.path.isdir(os.path.join(root_dir, "rv_label", "train")) or
        os.path.isdir(os.path.join(root_dir, "rv_label", "test"))
    )

    if has_split_dirs:
        for split in ["train", "test"]:
            ids, faz_map, rv_map = detect_split_pairs(root_dir, split)
            if not ids:
                print(f"[WARN] {scan_type} split={split} 没有找到可分类的成对 mask")
                continue

            rows, failed = classify_one_group(
                split_name=split,
                ids=ids,
                faz_map=faz_map,
                rv_map=rv_map,
                metrics_map=metrics_map,
                label_map=label_map,
                api_url=api_url,
                scan_type=scan_type,
            )
            all_rows.extend(rows)
            all_failed.extend(failed)
    else:
        ids, faz_map, rv_map = detect_flat_pairs(root_dir)
        if not ids:
            raise RuntimeError(f"{scan_type} 没有找到可分类的成对 mask，请检查 faz_label / rv_label 目录结构")

        rows, failed = classify_one_group(
            split_name="all",
            ids=ids,
            faz_map=faz_map,
            rv_map=rv_map,
            metrics_map=metrics_map,
            label_map=label_map,
            api_url=api_url,
            scan_type=scan_type,
        )
        all_rows.extend(rows)
        all_failed.extend(failed)

    return all_rows, all_failed


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(description="同时批量处理 output/3m 和 output/6m，并与标签表比较准确率")
    parser.add_argument("--output_root", type=str, required=True, help="如 /mnt/d/octa_agent/backend/output")
    parser.add_argument("--labels_xlsx", type=str, required=True, help="如 /mnt/data/Textlabels.xlsx")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/api/v1/agent/classify")
    parser.add_argument("--save_dir", type=str, default=None, help="结果保存目录，默认 output_root/classify_eval_results")
    args = parser.parse_args()

    output_root = args.output_root
    dir_3m = os.path.join(output_root, "3m")
    dir_6m = os.path.join(output_root, "6m")

    metrics_3m = os.path.join(dir_3m, "metrics_3m.xlsx")
    metrics_6m = os.path.join(dir_6m, "metrics_6m.xlsx")

    if not os.path.isdir(dir_3m):
        raise FileNotFoundError(f"找不到目录: {dir_3m}")
    if not os.path.isdir(dir_6m):
        raise FileNotFoundError(f"找不到目录: {dir_6m}")
    if not os.path.isfile(metrics_3m):
        raise FileNotFoundError(f"找不到文件: {metrics_3m}")
    if not os.path.isfile(metrics_6m):
        raise FileNotFoundError(f"找不到文件: {metrics_6m}")

    save_dir = args.save_dir or os.path.join(output_root, "classify_eval_results")
    os.makedirs(save_dir, exist_ok=True)

    label_map, used_id_col, used_label_col = load_labels_map(args.labels_xlsx, id_col=None, label_col=None)
    print(f"labels_xlsx  : {args.labels_xlsx}")
    print(f"label id col : {used_id_col}")
    print(f"label col    : {used_label_col}")
    print(f"api_url      : {args.api_url}")
    print(f"save_dir     : {save_dir}")

    all_rows = []
    all_failed = []

    # 处理 3M
    rows_3m, failed_3m = process_one_scan_root(
        root_dir=dir_3m,
        scan_type="3M",
        api_url=args.api_url,
        metrics_xlsx=metrics_3m,
        label_map=label_map,
    )
    all_rows.extend(rows_3m)
    all_failed.extend(failed_3m)

    # 处理 6M
    rows_6m, failed_6m = process_one_scan_root(
        root_dir=dir_6m,
        scan_type="6M",
        api_url=args.api_url,
        metrics_xlsx=metrics_6m,
        label_map=label_map,
    )
    all_rows.extend(rows_6m)
    all_failed.extend(failed_6m)

    if not all_rows:
        raise RuntimeError("没有得到任何分类结果，请检查目录结构或接口状态")

    df_pred = pd.DataFrame(all_rows).sort_values(by=["scan_type", "split", "image_id"])
    pred_path = os.path.join(save_dir, "predictions_all.xlsx")
    df_pred.to_excel(pred_path, index=False)

    df_eval = evaluate_predictions(df_pred)
    eval_path = os.path.join(save_dir, "evaluation_summary.xlsx")
    df_eval.to_excel(eval_path, index=False)

    save_confusion_matrices(df_pred, save_dir)

    if all_failed:
        df_failed = pd.DataFrame(all_failed)
        failed_path = os.path.join(save_dir, "failed_cases.xlsx")
        df_failed.to_excel(failed_path, index=False)
        print(f"⚠️ 失败记录已保存: {failed_path}")

    # 同时保存 json 版本
    summary_json = {
        "label_id_col": used_id_col,
        "label_col": used_label_col,
        "num_predictions": int(len(df_pred)),
        "num_failed": int(len(all_failed)),
        "metrics": df_eval.to_dict(orient="records"),
    }
    json_path = os.path.join(save_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"✅ 预测结果已保存: {pred_path}")
    print(f"✅ 评估汇总已保存: {eval_path}")
    print(f"✅ JSON汇总已保存: {json_path}")
    print("\n=== Evaluation Summary ===")
    print(df_eval.to_string(index=False))


if __name__ == "__main__":
    main()