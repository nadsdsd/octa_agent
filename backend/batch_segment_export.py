import os
import io
import base64
import argparse
from pathlib import Path

import requests
import pandas as pd
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def decode_dataurl_to_image(data_url: str) -> Image.Image:
    """
    将 data:image/png;base64,... 解码为 PIL Image
    """
    if "," in data_url:
        _, b64_data = data_url.split(",", 1)
    else:
        b64_data = data_url
    img_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(img_bytes)).convert("L")


def save_mask_from_base64(mask_data_url: str, save_path: str):
    img = decode_dataurl_to_image(mask_data_url)
    img.save(save_path)


def collect_images(input_dir: str, recursive: bool = True):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    input_path = Path(input_dir)

    if recursive:
        image_paths = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        image_paths = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

    return sorted(image_paths)


def call_segmentation_api(image_path: Path, api_url: str, timeout: int = 120):
    with open(image_path, "rb") as f:
        files = {
            "file": (image_path.name, f, "application/octet-stream")
        }
        response = requests.post(api_url, files=files, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    result = response.json()
    if result.get("status") != "success":
        raise RuntimeError(f"API返回失败: {result}")

    return result


def main():
    parser = argparse.ArgumentParser(description="批量调用本地OCTA分割后端，导出mask和指标")
    parser.add_argument("--input_dir", type=str, default="/mnt/d/finaldata/dataset1")
    parser.add_argument("--output_dir", type=str, default="/mnt/d/octa_agent/backend/output")
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/api/v1/agent/vision/analyze",
        help="分割接口地址"
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="是否递归遍历子目录")
    args = parser.parse_args()

    image_paths = collect_images(args.input_dir, recursive=args.recursive)

    if not image_paths:
        print(f"未找到可处理图片: {args.input_dir}")
        return

    # 输出目录
    out_3m_rv = os.path.join(args.output_dir, "3m", "rv_label")
    out_3m_faz = os.path.join(args.output_dir, "3m", "faz_label")
    out_6m_rv = os.path.join(args.output_dir, "6m", "rv_label")
    out_6m_faz = os.path.join(args.output_dir, "6m", "faz_label")

    ensure_dir(out_3m_rv)
    ensure_dir(out_3m_faz)
    ensure_dir(out_6m_rv)
    ensure_dir(out_6m_faz)

    rows_3m = []
    rows_6m = []
    failed_rows = []

    total = len(image_paths)
    print(f"共发现 {total} 张图片，开始处理...")

    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{total}] {image_path}")

        try:
            result = call_segmentation_api(image_path, args.api_url)

            image_metadata = result.get("image_metadata", {})
            metrics = result.get("metrics", {})
            visualizations = result.get("visualizations", {})

            scan_type = str(image_metadata.get("scan_type", "")).upper().strip()
            rv_mask_base64 = visualizations.get("rv_mask_base64")
            faz_mask_base64 = visualizations.get("faz_mask_base64")

            if not rv_mask_base64 or not faz_mask_base64:
                raise RuntimeError("返回结果中缺少 rv_mask_base64 或 faz_mask_base64")

            # image_id：例如 10001.bmp -> 10001
            image_id = image_path.stem

            # 保存mask
            if scan_type == "3M":
                rv_save_path = os.path.join(out_3m_rv, f"{image_id}.png")
                faz_save_path = os.path.join(out_3m_faz, f"{image_id}.png")
            elif scan_type == "6M":
                rv_save_path = os.path.join(out_6m_rv, f"{image_id}.png")
                faz_save_path = os.path.join(out_6m_faz, f"{image_id}.png")
            else:
                raise RuntimeError(f"未知scan_type: {scan_type}")

            save_mask_from_base64(rv_mask_base64, rv_save_path)
            save_mask_from_base64(faz_mask_base64, faz_save_path)

            # Excel里只保留 image_id + 各项指标
            row = {
                "image_id": image_id
            }
            for k, v in metrics.items():
                row[k] = v

            if scan_type == "3M":
                rows_3m.append(row)
            else:
                rows_6m.append(row)

        except Exception as e:
            print(f"❌ 处理失败: {image_path.name} -> {e}")
            failed_rows.append({
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "file_path": str(image_path),
                "error": str(e)
            })

    # 导出Excel
    if rows_3m:
        df_3m = pd.DataFrame(rows_3m)
        df_3m = df_3m.sort_values(by="image_id")
        save_path_3m = os.path.join(args.output_dir, "3m", "metrics_3m.xlsx")
        df_3m.to_excel(save_path_3m, index=False)
        print(f"✅ 3M指标表已保存: {save_path_3m}")
    else:
        print("⚠️ 没有生成任何3M结果")

    if rows_6m:
        df_6m = pd.DataFrame(rows_6m)
        df_6m = df_6m.sort_values(by="image_id")
        save_path_6m = os.path.join(args.output_dir, "6m", "metrics_6m.xlsx")
        df_6m.to_excel(save_path_6m, index=False)
        print(f"✅ 6M指标表已保存: {save_path_6m}")
    else:
        print("⚠️ 没有生成任何6M结果")

    if failed_rows:
        df_failed = pd.DataFrame(failed_rows)
        failed_path = os.path.join(args.output_dir, "failed_cases.xlsx")
        df_failed.to_excel(failed_path, index=False)
        print(f"⚠️ 失败记录已保存: {failed_path}")

    print("处理完成。")


if __name__ == "__main__":
    main()