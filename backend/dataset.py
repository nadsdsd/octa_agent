# # dataset.py
# import os, re, glob
# from typing import List, Dict, Optional, Tuple
# import numpy as np
# import pandas as pd
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

# TARGET_SIZE = (224, 224)

# def _id_from_name(name: str) -> Optional[int]:
#     m = re.search(r"(\d{5})", name)
#     return int(m.group(1)) if m else None

# def _read_mask_224(path: str) -> torch.Tensor:
#     """灰度读取 -> 最近邻到 224×224 -> 二值（白=1） -> float32 张量 (H,W)"""
#     img = Image.open(path).convert("L")
#     if img.size != TARGET_SIZE:
#         img = img.resize(TARGET_SIZE, Image.NEAREST)
#     u8 = np.array(img)
#     if np.unique(u8).size <= 4:
#         mask = (u8 > 0)
#     else:
#         mask = (u8 >= 128)
#     return torch.from_numpy(mask.astype(np.float32))

# def _scan_split_one_root(root: str, split: str) -> Dict[int, Tuple[str, str]]:
#     """
#     返回 {编号: (faz_path, rv_path)}；root 是 data_3m 或 data_6m
#     只收集同时存在 FAZ 与 RV 的编号
#     """
#     out = {}
#     faz_dir = os.path.join(root, "FAZ", split)
#     rv_dir  = os.path.join(root, "RV",  split)
#     faz_files = glob.glob(os.path.join(faz_dir, "**", "*.bmp"), recursive=True) + \
#                 glob.glob(os.path.join(faz_dir, "**", "*.BMP"), recursive=True)
#     rv_files  = glob.glob(os.path.join(rv_dir,  "**", "*.bmp"), recursive=True) + \
#                 glob.glob(os.path.join(rv_dir,  "**", "*.BMP"), recursive=True)
#     faz_map = {_id_from_name(os.path.basename(p)): p for p in faz_files if _id_from_name(os.path.basename(p))}
#     rv_map  = {_id_from_name(os.path.basename(p)): p for p in rv_files  if _id_from_name(os.path.basename(p))}
#     for k in sorted(set(faz_map) & set(rv_map)):
#         if 10001 <= k <= 10500:
#             out[k] = (faz_map[k], rv_map[k])
#     return out

# def fov_from_id(img_id: int) -> int:
#     """
#     根据编号末三位判断 FOV：
#       001–300 -> 6mm（返回 1）
#       301–500 -> 3mm（返回 0）
#     """
#     tail = img_id % 1000
#     return 1 if 1 <= tail <= 300 else 0  # 1:6m, 0:3m

# def load_labels_multi(label_xlsx: str, label_col: str = "Disease") -> Tuple[Dict[int, int], LabelEncoder, List[str]]:
#     """
#     多类别标签：返回 (id->class_index, label_encoder, class_names)
#     """
#     df = pd.read_excel(label_xlsx)
#     if "编号" not in df.columns or label_col not in df.columns:
#         raise ValueError(f"标签文件必须包含列: 编号 和 {label_col}")
#     df = df.dropna(subset=["编号", label_col])
#     df["编号"] = df["编号"].astype(int)

#     le = LabelEncoder()
#     y = le.fit_transform(df[label_col].astype(str).values)  # 多类别编码
#     id2y = {int(i): int(c) for i, c in zip(df["编号"].values, y)}
#     return id2y, le, list(le.classes_)

# def load_metrics(metrics_xlsx: str) -> pd.DataFrame:
#     df = pd.read_excel(metrics_xlsx, sheet_name=0)
#     need = ["编号","faz_area_px","faz_几何_circularity","rv_密度","rv_分支数量"]
#     miss = [c for c in need if c not in df.columns]
#     if miss:
#         raise ValueError(f"指标文件缺少列: {miss}")
#     return df[need].copy()

# class OCTADataset(Dataset):
#     """
#     mode = "img" 或 "img+metrics"
#     图像输入为 2 通道：[RV, FAZ]  (2,224,224)
#     额外返回 id 与 fov（0=3m, 1=6m）
#     """
#     def __init__(
#         self,
#         data_roots: List[str],   # 例如 ["./data_3m", "./data_6m"]
#         split: str,              # "train"/"val"/"test"
#         id2y: Dict[int,int],
#         mode: str = "img",
#         metrics_df: Optional[pd.DataFrame] = None,
#         metrics_norm: Optional[Dict[str, Tuple[float,float]]] = None,
#     ):
#         assert mode in ("img","img+metrics")
#         self.mode = mode

#         found = {}
#         for r in data_roots:
#             if not os.path.isdir(r): 
#                 continue
#             sub = _scan_split_one_root(r, split)
#             found.update(sub)

#         items = []
#         for k,(fz,rv) in found.items():
#             if k in id2y:
#                 items.append((k,fz,rv,id2y[k], fov_from_id(k)))
#         self.items = sorted(items, key=lambda x: x[0])

#         # 指标
#         self.metrics = None
#         self.metrics_cols = ["faz_area_px","faz_几何_circularity","rv_密度","rv_分支数量"]
#         if self.mode == "img+metrics":
#             assert metrics_df is not None, "img+metrics 模式需要提供 metrics_xlsx"
#             m = metrics_df.set_index("编号").reindex([k for (k,_,_,_,_) in self.items])[self.metrics_cols]
#             if metrics_norm is not None:
#                 for c, (mu, sigma) in metrics_norm.items():
#                     if sigma > 0:
#                         m[c] = (m[c] - mu) / sigma
#             self.metrics = m.fillna(0.0).astype(np.float32).values

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, idx):
#         k, faz_p, rv_p, y, fov = self.items[idx]
#         faz = _read_mask_224(faz_p)  # (H,W)
#         rv  = _read_mask_224(rv_p)
#         img = torch.stack([rv, faz], dim=0)  # (2,224,224)
#         label = torch.tensor(y, dtype=torch.long)
#         fov_t = torch.tensor(fov, dtype=torch.long)
#         if self.mode == "img":
#             return img, label, k, fov_t
#         m = torch.from_numpy(self.metrics[idx])
#         return img, label, k, fov_t, m

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# import os, re, glob
# from typing import List, Dict, Optional, Tuple
# import numpy as np
# import pandas as pd
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

# TARGET_SIZE = (224, 224)

# # ===================== 基础工具 =====================

# def _read_mask_224(path: str) -> torch.Tensor:
#     """灰度读取 -> 最近邻到 224×224 -> 二值（白=1） -> float32 张量 (H,W)"""
#     img = Image.open(path).convert("L")
#     if img.size != TARGET_SIZE:
#         img = img.resize(TARGET_SIZE, Image.NEAREST)
#     u8 = np.array(img)
#     if np.unique(u8).size <= 4:
#         mask = (u8 > 0)
#     else:
#         mask = (u8 >= 128)
#     return torch.from_numpy(mask.astype(np.float32))

# def _extract_id_token(basename: str) -> Optional[str]:
#     """提取 ID：优先取文件名第一个 '_' 前的 token，否则取去后缀的整名。"""
#     name = os.path.splitext(basename)[0]
#     if "_" in name: tok = name.split("_")[0]
#     else: tok = name
#     tok = tok.strip()
#     return tok if tok else None

# def _glob_exts(d):
#     exts = ["png","jpg","jpeg","bmp","tif","tiff","PNG","JPG","JPEG","BMP","TIF","TIFF"]
#     out = []
#     for e in exts:
#         out += glob.glob(os.path.join(d, f"**/*.{e}"), recursive=True)
#     return out

# def _scan_split_one_root(root: str, split: str) -> Dict[str, Tuple[str, str, str]]:
#     """
#     返回 {id_token: (faz_path, rv_path, root_path)}
#     优先双文件夹 root/FAZ/split & root/RV/split；否则单文件夹 root/split。
#     """
#     out: Dict[str, Tuple[str,str,str]] = {}

#     # --- 方案1: 双文件夹 ---
#     faz_dir = os.path.join(root, "FAZ", split)
#     rv_dir  = os.path.join(root, "RV",  split)
#     if os.path.isdir(faz_dir) and os.path.isdir(rv_dir):
#         faz_files = _glob_exts(faz_dir)
#         rv_files  = _glob_exts(rv_dir)

#         def _make_map(files, expect_flag: Optional[str]):
#             m = {}
#             for p in files:
#                 bn = os.path.basename(p)
#                 key = _extract_id_token(bn)
#                 if not key: continue
#                 if expect_flag is not None and expect_flag.lower() not in bn.lower():
#                     continue
#                 m.setdefault(key, []).append(p)
#             m1 = {}
#             for k, arr in m.items():
#                 arr = sorted(arr, key=lambda x: (len(os.path.basename(x)), x))
#                 m1[k] = arr[0]
#             return m1

#         faz_map = _make_map(faz_files, "faz") or _make_map(faz_files, None)
#         rv_map  = _make_map(rv_files,  "rv")  or _make_map(rv_files,  None)

#         for k in set(faz_map) & set(rv_map):
#             out[k] = (faz_map[k], rv_map[k], root)
#         if out:
#             return out

#     # --- 方案2: 单文件夹 ---
#     split_dir = os.path.join(root, split)
#     if os.path.isdir(split_dir):
#         files = _glob_exts(split_dir)
#         faz_map, rv_map = {}, {}
#         for p in files:
#             bn = os.path.basename(p)
#             key = _extract_id_token(bn)
#             if not key: continue
#             low = bn.lower()
#             if "faz" in low: faz_map.setdefault(key, p)
#             elif "rv" in low: rv_map.setdefault(key, p)

#         # 正常：有 faz/rv 标记
#         for k in set(faz_map) & set(rv_map):
#             out[k] = (faz_map[k], rv_map[k], root)

#         # 兜底：同 ID 下至少两个文件，前者当 faz、后者 rv（尽量不用）
#         if not out:
#             from collections import defaultdict
#             grp = defaultdict(list)
#             for p in files:
#                 bn = os.path.basename(p)
#                 key = _extract_id_token(bn)
#                 if key: grp[key].append(p)
#             for k, arr in grp.items():
#                 if len(arr) >= 2:
#                     arr = sorted(arr)
#                     out[k] = (arr[0], arr[1], root)

#     return out

# def fov_from_id_or_root(img_id_token: str, root_path: str) -> int:
#     """
#     FOV 推断：
#       - 根目录名含 '3' => 3mm(0)，含 '6' => 6mm(1)
#       - 否则回退：抽取数字末三位；1..300 -> 6mm(1)，其它 -> 3mm(0)；再不行默认 3mm(0)
#     """
#     base = os.path.basename(os.path.abspath(root_path)).lower()
#     if "3" in base: return 0
#     if "6" in base: return 1
#     try:
#         as_int = int(re.sub(r"\D", "", img_id_token))
#         tail = as_int % 1000
#         return 1 if 1 <= tail <= 300 else 0
#     except Exception:
#         return 0

# # ===================== Excel 读取 =====================

# def load_labels_multi(label_xlsx: str, label_col: str = "Disease") -> Tuple[Dict[str, int], LabelEncoder, List[str]]:
#     """
#     多类别标签：返回 (id_token[str] -> class_index, LabelEncoder, class_names)
#     自动识别 ID 列：优先 '编号' / 用户传入，再尝试常见列名。
#     """
#     df = pd.read_excel(label_xlsx)
#     df.columns = [str(c).strip() for c in df.columns]
#     candidates = ["编号","ID","Id","id","case_id","CaseID","Subject","PatientID"]
#     id_col = None
#     for c in candidates:
#         if c in df.columns:
#             id_col = c; break
#     if id_col is None or label_col not in df.columns:
#         raise ValueError(f"标签文件需要包含 ID 列({candidates}) 和 {label_col}")

#     df = df.dropna(subset=[id_col, label_col]).copy()

#     # 统一 ID 格式为字符串
#     def _to_key(v):
#         if isinstance(v, (int, np.integer, float, np.floating)) and float(v).is_integer():
#             return str(int(v))
#         return str(v)
#     df[id_col] = df[id_col].map(_to_key)

#     le = LabelEncoder()
#     y = le.fit_transform(df[label_col].astype(str).values)
#     id2y = {str(i): int(c) for i, c in zip(df[id_col].values, y)}
#     return id2y, le, list(le.classes_)

# def _row_to_text(row: pd.Series, id_col: str) -> str:
#     parts = []
#     for col, val in row.items():
#         if col == id_col or pd.isna(val): continue
#         sval = f"{float(val):.6g}" if isinstance(val, (float, np.floating)) else str(val)
#         parts.append(f"{col}: {sval}")
#     return "; ".join(parts) if parts else "no_features"

# def load_tabletext_sheets(excel_path: str,
#                           sheet_3m: str = "3M_metrics",
#                           sheet_6m: str = "6M_metrics",
#                           id_col: str = "编号") -> Tuple[Dict[str,str], Dict[str,str]]:
#     """把每行表格转 'col: val; ...' 文本；输出两个 sheet 的 {id_token: text} 映射。"""
#     def _read_sheet(name):
#         df = pd.read_excel(excel_path, sheet_name=name)
#         df.columns = [str(c).strip() for c in df.columns]
#         return df

#     df3 = _read_sheet(sheet_3m)
#     df6 = _read_sheet(sheet_6m)

#     def _auto_id(df, preferred):
#         if preferred in df.columns: return preferred
#         cands = ["编号","ID","Id","id","case_id","CaseID","Subject","PatientID"]
#         for c in cands:
#             if c in df.columns: return c
#         raise ValueError(f"[{excel_path}] 无法识别 ID 列；请用 --id_col 指定")

#     id3 = _auto_id(df3, id_col)
#     id6 = _auto_id(df6, id_col)

#     def _norm_id(x):
#         if isinstance(x, (int, np.integer, float, np.floating)) and float(x).is_integer():
#             return str(int(x))
#         return str(x)

#     for df, ic in ((df3,id3),(df6,id6)):
#         df.dropna(axis=1, how="all", inplace=True)
#         df.dropna(how="all", inplace=True)
#         df.drop_duplicates(subset=[ic], inplace=True)
#         df[ic] = df[ic].map(_norm_id)

#     text3 = {row[id3]: _row_to_text(row, id3) for _, row in df3.iterrows()}
#     text6 = {row[id6]: _row_to_text(row, id6) for _, row in df6.iterrows()}
#     return text3, text6

# def load_metrics(metrics_xlsx: str) -> pd.DataFrame:
#     """用于 img+metrics（数值指标）——示例四列，可按需修改为你的实际列"""
#     df = pd.read_excel(metrics_xlsx, sheet_name=0)
#     need = ["编号","faz_area_px","faz_几何_circularity","rv_密度","rv_分支数量"]
#     miss = [c for c in need if c not in df.columns]
#     if miss:
#         raise ValueError(f"指标文件缺少列: {miss}")
#     return df[need].copy()

# # ===================== 数据集 =====================

# class OCTADataset(Dataset):
#     """
#     支持三种模式：
#       - "img"           : 仅图像（2通道: [RV, FAZ]，224×224）
#       - "img+metrics"   : 图像 + 数值指标（默认 4 维，见 load_metrics）
#       - "img+tabletext" : 图像 + 文本（来自 Excel 的 3M/6M sheet 序列化）
#     返回：img, label, id_token(str), fov(0/1), [metrics | text]
#     """
#     def __init__(
#         self,
#         data_roots: List[str],
#         split: str,  # "train"/"val"/"test"
#         id2y: Dict[str,int],
#         mode: str = "img",
#         metrics_df: Optional[pd.DataFrame] = None,
#         metrics_norm: Optional[Dict[str, Tuple[float,float]]] = None,
#         tabletext_3m: Optional[Dict[str,str]] = None,
#         tabletext_6m: Optional[Dict[str,str]] = None,
#     ):
#         assert mode in ("img","img+metrics","img+tabletext")
#         self.mode = mode

#         found: Dict[str, Tuple[str,str,str]] = {}
#         for r in data_roots:
#             if not os.path.isdir(r): 
#                 continue
#             sub = _scan_split_one_root(r, split)
#             found.update(sub)  # 同 ID 后者覆盖

#         items = []
#         for k,(fz,rv,root_path) in found.items():
#             # 先直接字符串匹配；再尝试提取数字部分与标签表匹配
#             if k in id2y:
#                 y = id2y[k]
#             else:
#                 digits = re.sub(r"\D", "", k)
#                 if digits and digits in id2y:
#                     y = id2y[digits]
#                 else:
#                     continue
#             fov = fov_from_id_or_root(k, root_path)
#             items.append((k, fz, rv, y, fov))
#         self.items = sorted(items, key=lambda x: str(x[0]))

#         # 数值指标
#         self.metrics = None
#         self.metrics_cols = ["faz_area_px","faz_几何_circularity","rv_密度","rv_分支数量"]
#         if self.mode == "img+metrics":
#             assert metrics_df is not None, "img+metrics 模式需要提供 metrics_xlsx"
#             m = metrics_df.set_index("编号").reindex(
#                 [re.sub(r'\\D','', str(k)) if str(k) not in metrics_df["编号"].astype(str).values else str(k)
#                  for (k,_,_,_,_) in self.items]
#             )[self.metrics_cols]
#             if metrics_norm is not None:
#                 for c, (mu, sigma) in metrics_norm.items():
#                     if sigma > 0: m[c] = (m[c] - mu) / sigma
#             self.metrics = m.fillna(0.0).astype(np.float32).values

#         # 文本映射（3M/6M）
#         self.text_3m = tabletext_3m if self.mode == "img+tabletext" else None
#         self.text_6m = tabletext_6m if self.mode == "img+tabletext" else None
#         if self.mode == "img+tabletext":
#             if self.text_3m is None or self.text_6m is None:
#                 raise ValueError("img+tabletext 模式需要提供两个 sheet 的文本映射（3M/6M）。")

#     def __len__(self): return len(self.items)

#     def __getitem__(self, idx):
#         k, faz_p, rv_p, y, fov = self.items[idx]
#         faz = _read_mask_224(faz_p)  # (H,W)
#         rv  = _read_mask_224(rv_p)
#         img = torch.stack([rv, faz], dim=0)  # (2,224,224)
#         label = torch.tensor(y, dtype=torch.long)
#         fov_t = torch.tensor(fov, dtype=torch.long)

#         if self.mode == "img":
#             return img, label, k, fov_t

#         if self.mode == "img+metrics":
#             m = torch.from_numpy(self.metrics[idx])
#             return img, label, k, fov_t, m

#         # img+tabletext
#         text = (self.text_3m.get(str(k), "no_features") if fov == 0
#                 else self.text_6m.get(str(k), "no_features"))
#         return img, label, k, fov_t, text
# ✅ Full replacements

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os, re, glob
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

TARGET_SIZE = (224, 224)

# ===================== 基础工具 =====================
def _read_mask_224(path: str) -> torch.Tensor:
    """灰度读取 -> 最近邻到 224×224 -> 二值（白=1） -> float32 张量 (H,W)"""
    img = Image.open(path).convert("L")
    if img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.NEAREST)
    u8 = np.array(img)
    if np.unique(u8).size <= 4:
        mask = (u8 > 0)
    else:
        mask = (u8 >= 128)
    return torch.from_numpy(mask.astype(np.float32))

def _extract_id_token(basename: str) -> Optional[str]:
    """提取 ID：优先取文件名第一个 '_' 前的 token，否则取去后缀的整名。"""
    name = os.path.splitext(basename)[0]
    if "_" in name:
        tok = name.split("_")[0]
    else:
        tok = name
    tok = tok.strip()
    return tok if tok else None

def _glob_exts(d):
    exts = ["png","jpg","jpeg","bmp","tif","tiff","PNG","JPG","JPEG","BMP","TIF","TIFF"]
    out = []
    for e in exts:
        out += glob.glob(os.path.join(d, f"**/*.{e}"), recursive=True)
    return out

def _scan_split_one_root(root: str, split: str) -> Dict[str, Tuple[str, str, str]]:
    """
    返回 {id_token: (faz_path, rv_path, root_path)}
    兼容以下目录名：
      - root/FAZ/train, root/RV/train
      - root/faz_label/train, root/rv_label/train
      - root/faz/train, root/rv/train
    """
    out: Dict[str, Tuple[str, str, str]] = {}

    faz_aliases = ["FAZ", "faz_label", "faz", "Faz"]
    rv_aliases  = ["RV", "rv_label", "rv", "Rv"]

    def _find_view_dir(root_dir: str, aliases: List[str], split_name: str) -> Optional[str]:
        # 优先找 root/<alias>/<split>
        for a in aliases:
            p = os.path.join(root_dir, a, split_name)
            if os.path.isdir(p):
                return p

        # 再找 root/<alias>
        for a in aliases:
            p = os.path.join(root_dir, a)
            if os.path.isdir(p):
                return p

        return None

    faz_dir = _find_view_dir(root, faz_aliases, split)
    rv_dir  = _find_view_dir(root, rv_aliases, split)

    # --- 方案1：双文件夹 ---
    if faz_dir is not None and rv_dir is not None:
        faz_files = _glob_exts(faz_dir)
        rv_files  = _glob_exts(rv_dir)

        def _make_map(files, expect_flag: Optional[str]):
            m = {}
            for p in files:
                bn = os.path.basename(p)
                key = _extract_id_token(bn)
                if not key:
                    continue
                if expect_flag is not None and expect_flag.lower() not in bn.lower():
                    continue
                m.setdefault(key, []).append(p)

            m1 = {}
            for k, arr in m.items():
                arr = sorted(arr, key=lambda x: (len(os.path.basename(x)), x))
                m1[k] = arr[0]
            return m1

        # 先尝试文件名里带 faz/rv 标记；没有就直接按 id 对齐
        faz_map = _make_map(faz_files, "faz") or _make_map(faz_files, None)
        rv_map  = _make_map(rv_files,  "rv")  or _make_map(rv_files,  None)

        for k in set(faz_map) & set(rv_map):
            out[k] = (faz_map[k], rv_map[k], root)

        if out:
            return out

    # --- 方案2：单文件夹 root/split，文件名里区分 faz/rv ---
    split_dir = os.path.join(root, split)
    if os.path.isdir(split_dir):
        files = _glob_exts(split_dir)
        faz_map, rv_map = {}, {}

        for p in files:
            bn = os.path.basename(p)
            key = _extract_id_token(bn)
            if not key:
                continue
            low = bn.lower()
            if "faz" in low:
                faz_map.setdefault(key, p)
            elif "rv" in low:
                rv_map.setdefault(key, p)

        for k in set(faz_map) & set(rv_map):
            out[k] = (faz_map[k], rv_map[k], root)

        if not out:
            from collections import defaultdict
            grp = defaultdict(list)
            for p in files:
                bn = os.path.basename(p)
                key = _extract_id_token(bn)
                if key:
                    grp[key].append(p)
            for k, arr in grp.items():
                if len(arr) >= 2:
                    arr = sorted(arr)
                    out[k] = (arr[0], arr[1], root)

    return out

def fov_from_id_or_root(img_id_token: str, root_path: str) -> int:
    """
    FOV 推断：
      - 根目录名含 '3' => 3mm(0)，含 '6' => 6mm(1)
      - 否则回退：抽取数字末三位；1..300 -> 6mm(1)，其它 -> 3mm(0)；再不行默认 3mm(0)
    """
    base = os.path.basename(os.path.abspath(root_path)).lower()
    if "3" in base:
        return 0
    if "6" in base:
        return 1
    try:
        as_int = int(re.sub(r"\D", "", img_id_token))
        tail = as_int % 1000
        return 1 if 1 <= tail <= 300 else 0
    except Exception:
        return 0

# ===================== Excel 读取 =====================
def load_labels_multi(label_xlsx: str, label_col: str = "Disease") -> Tuple[Dict[str, int], LabelEncoder, List[str]]:
    """
    多类别标签：返回 (id_token[str] -> class_index, LabelEncoder, class_names)
    自动识别 ID 列：优先 '编号' / 用户传入，再尝试常见列名。
    """
    df = pd.read_excel(label_xlsx)
    df.columns = [str(c).strip() for c in df.columns]
    candidates = ["编号","ID","Id","id","case_id","CaseID","Subject","PatientID"]
    id_col = None
    for c in candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None or label_col not in df.columns:
        raise ValueError(f"标签文件需要包含 ID 列({candidates}) 和 {label_col}")

    df = df.dropna(subset=[id_col, label_col]).copy()

    # 统一 ID 格式为字符串
    def _to_key(v):
        if isinstance(v, (int, np.integer, float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)
    df[id_col] = df[id_col].map(_to_key)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str).values)
    id2y = {str(i): int(c) for i, c in zip(df[id_col].values, y)}
    return id2y, le, list(le.classes_)

def _row_to_text(row: pd.Series, id_col: str) -> str:
    parts = []
    for col, val in row.items():
        if col == id_col or pd.isna(val):
            continue
        sval = f"{float(val):.6g}" if isinstance(val, (float, np.floating)) else str(val)
        parts.append(f"{col}: {sval}")
    return "; ".join(parts) if parts else "no_features"

def load_tabletext_sheets(excel_path: str,
                          sheet_3m: str = "3M_metrics",
                          sheet_6m: str = "6M_metrics",
                          id_col: str = "编号") -> Tuple[Dict[str,str], Dict[str,str]]:
    """把每行表格转 'col: val; ...' 文本；输出两个 sheet 的 {id_token: text} 映射。"""
    def _read_sheet(name):
        df = pd.read_excel(excel_path, sheet_name=name)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    df3 = _read_sheet(sheet_3m)
    df6 = _read_sheet(sheet_6m)

    def _auto_id(df, preferred):
        if preferred in df.columns:
            return preferred
        cands = ["编号","ID","Id","id","case_id","CaseID","Subject","PatientID"]
        for c in cands:
            if c in df.columns:
                return c
        raise ValueError(f"[{excel_path}] 无法识别 ID 列；请用 --id_col 指定")

    id3 = _auto_id(df3, id_col)
    id6 = _auto_id(df6, id_col)

    def _norm_id(x):
        if isinstance(x, (int, np.integer, float, np.floating)) and float(x).is_integer():
            return str(int(x))
        return str(x)

    for df, ic in ((df3,id3),(df6,id6)):
        df.dropna(axis=1, how="all", inplace=True)
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(subset=[ic], inplace=True)
        df[ic] = df[ic].map(_norm_id)

    text3 = {row[id3]: _row_to_text(row, id3) for _, row in df3.iterrows()}
    text6 = {row[id6]: _row_to_text(row, id6) for _, row in df6.iterrows()}
    return text3, text6

def load_metrics(metrics_xlsx: str) -> pd.DataFrame:
    """用于 img+metrics（数值指标）——示例四列，可按需修改为你的实际列"""
    df = pd.read_excel(metrics_xlsx, sheet_name=0)
    need = ["编号","faz_area_px","faz_几何_circularity","rv_密度","rv_分支数量"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"指标文件缺少列: {miss}")
    return df[need].copy()

# ===================== 数据集 =====================
class OCTADataset(Dataset):
    """
    支持三种模式：
      - "img"           : 仅图像（2通道: [RV, FAZ]，224×224）
      - "img+metrics"   : 图像 + 数值指标（默认 4 维，见 load_metrics）
      - "img+tabletext" : 图像 + 文本（来自 Excel 的 3M/6M sheet 序列化）
    返回：img, label, id_token(str), fov(0/1), [metrics | text]
    """
    def __init__(
        self,
        data_roots: List[str],
        split: str,  # "train"/"val"/"test"
        id2y: Dict[str,int],
        mode: str = "img",
        metrics_df: Optional[pd.DataFrame] = None,
        metrics_norm: Optional[Dict[str, Tuple[float,float]]] = None,
        tabletext_3m: Optional[Dict[str,str]] = None,
        tabletext_6m: Optional[Dict[str,str]] = None,
    ):
        assert mode in ("img","img+metrics","img+tabletext")
        self.mode = mode

        found: Dict[str, Tuple[str,str,str]] = {}
        for r in data_roots:
            if not os.path.isdir(r):
                continue
            sub = _scan_split_one_root(r, split)
            found.update(sub)  # 同 ID 后者覆盖

        items = []
        for k,(fz,rv,root_path) in found.items():
            # 先直接字符串匹配；再尝试提取数字部分与标签表匹配
            if k in id2y:
                y = id2y[k]
            else:
                digits = re.sub(r"\D", "", k)
                if digits and digits in id2y:
                    y = id2y[digits]
                else:
                    continue
            fov = fov_from_id_or_root(k, root_path)
            items.append((k, fz, rv, y, fov))
        self.items = sorted(items, key=lambda x: str(x[0]))

        # 数值指标
        self.metrics = None
        self.metrics_cols = ["faz_area_px","faz_几何_circularity","rv_密度","rv_分支数量"]
        if self.mode == "img+metrics":
            assert metrics_df is not None, "img+metrics 模式需要提供 metrics_xlsx"
            m = metrics_df.set_index("编号").reindex(
                [re.sub(r'\D','', str(k)) if str(k) not in metrics_df["编号"].astype(str).values else str(k)
                 for (k,_,_,_,_) in self.items]
            )[self.metrics_cols]
            if metrics_norm is not None:
                for c, (mu, sigma) in metrics_norm.items():
                    if sigma > 0:
                        m[c] = (m[c] - mu) / sigma
            self.metrics = m.fillna(0.0).astype(np.float32).values

        # 文本映射（3M/6M）
        self.text_3m = tabletext_3m if self.mode == "img+tabletext" else None
        self.text_6m = tabletext_6m if self.mode == "img+tabletext" else None
        if self.mode == "img+tabletext":
            if self.text_3m is None or self.text_6m is None:
                raise ValueError("img+tabletext 模式需要提供两个 sheet 的文本映射（3M/6M）。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        k, faz_p, rv_p, y, fov = self.items[idx]
        faz = _read_mask_224(faz_p)  # (H,W)
        rv  = _read_mask_224(rv_p)
        img = torch.stack([rv, faz], dim=0)  # (2,224,224)
        label = torch.tensor(y, dtype=torch.long)
        fov_t = torch.tensor(fov, dtype=torch.long)

        if self.mode == "img":
            return img, label, k, fov_t

        if self.mode == "img+metrics":
            m = torch.from_numpy(self.metrics[idx])
            return img, label, k, fov_t, m

        # img+tabletext
        text = (self.text_3m.get(str(k), "no_features") if fov == 0
                else self.text_6m.get(str(k), "no_features"))
        return img, label, k, fov_t, text
# import os
# import re
# from typing import List, Dict, Optional

# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms


# # -------------------- 基础工具 --------------------
# def _norm_ext(name: str):
#     return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


# def _extract_id_from_name(fname: str) -> str:
#     """
#     从文件名中稳健抽取 id：
#     1) 下划线/连字符前的 token
#     2) 不行则抽取连续数字
#     3) 否则整名去后缀
#     """
#     stem = os.path.splitext(os.path.basename(fname))[0]
#     if "_" in stem:
#         tok = stem.split("_")[0]
#         if tok:
#             return tok
#     if "-" in stem:
#         tok = stem.split("-")[0]
#         if tok:
#             return tok
#     m = re.search(r"\d+", stem)
#     if m:
#         return m.group(0)
#     return stem


# def _gather_split(root_dir: str, view: str) -> Dict[str, str]:
#     """
#     收集单一视野(view=RV/FAZ)下 train/val/test 三子目录的图片，取“最后一次出现”为该 id 的该视野路径。
#     返回：{ id -> 路径 }
#     """
#     id2path: Dict[str, str] = {}
#     for split in ["train", "val", "test"]:
#         d = os.path.join(root_dir, view, split)
#         if not os.path.isdir(d):
#             continue
#         for nm in os.listdir(d):
#             if not _norm_ext(nm):
#                 continue
#             p = os.path.join(d, nm)
#             iid = _extract_id_from_name(nm)
#             id2path[str(iid)] = p
#     return id2path


# # -------------------- 新：构建 RV+FAZ 成对索引 --------------------
# def build_pair_index(root_dir: str) -> List[Dict]:
#     """
#     为每个 id 汇总 RV 与 FAZ 的路径；允许缺一会用另一张图“复制”到两通道（保持 2ch 输入）。
#     返回元素：
#       {'id': '10001', 'rv': '/path/RV/..png' or None, 'faz': '/path/FAZ/..png' or None}
#     """
#     rv_map = _gather_split(root_dir, "RV")
#     fz_map = _gather_split(root_dir, "FAZ")

#     all_ids = sorted(set(rv_map.keys()) | set(fz_map.keys()))
#     pairs = []
#     for iid in all_ids:
#         pairs.append({
#             "id": iid,
#             "rv": rv_map.get(iid, None),
#             "faz": fz_map.get(iid, None),
#         })
#     return pairs


# # -------------------- 数据集：返回 2ch 图 + 文本 --------------------
# class MultiViewWithTextDataset(Dataset):
#     """
#     每个样本：
#       - image: Tensor[2,H,W]，通道0=RV灰度，通道1=FAZ灰度；若缺一则复制另一通道
#       - text:  由八项数值指标拼接成的字符串（供文本编码器）
#       - label: int
#       - id:    str
#     """
#     def __init__(self,
#                  pair_items: List[Dict],
#                  id2label: Dict[str, int],
#                  id2text: Dict[str, str],
#                  indices: Optional[List[int]] = None,
#                  image_size: int = 224):
#         self.items = pair_items if indices is None else [pair_items[i] for i in indices]
#         # 仅保留有标签的 id
#         self.items = [x for x in self.items if x["id"] in id2label]
#         self.id2label = id2label
#         self.id2text = id2text

#         # 统一预处理：转灰度 -> Resize -> ToTensor -> Normalize(2 通道)
#         self.resize = transforms.Resize((image_size, image_size))
#         self.to_tensor = transforms.ToTensor()
#         self.norm = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])  # 简单 0.5/0.5 归一

#     def __len__(self):
#         return len(self.items)

#     def _load_gray(self, path: str):
#         img = Image.open(path).convert("L")  # 灰度
#         img = self.resize(img)
#         return self.to_tensor(img)  # [1,H,W], 0~1

#     def __getitem__(self, idx):
#         rec = self.items[idx]
#         iid = rec["id"]

#         # 读两视野；若缺一，则把另一通道复制两份，保证 2ch
#         rv_t, fz_t = None, None
#         if rec["rv"] and os.path.isfile(rec["rv"]):
#             rv_t = self._load_gray(rec["rv"])  # [1,H,W]
#         if rec["faz"] and os.path.isfile(rec["faz"]):
#             fz_t = self._load_gray(rec["faz"])

#         if rv_t is None and fz_t is None:
#             raise FileNotFoundError(f"id={iid} 的 RV/FAZ 图都不存在：{rec}")

#         if rv_t is None:
#             rv_t = fz_t.clone()
#         if fz_t is None:
#             fz_t = rv_t.clone()

#         img2 = torch.cat([rv_t, fz_t], dim=0)  # [2,H,W]
#         img2 = self.norm(img2)

#         text = self.id2text.get(iid, "")  # 允许空串
#         label = int(self.id2label[iid])

#         return {"image": img2, "label": label, "id": iid, "text": text}