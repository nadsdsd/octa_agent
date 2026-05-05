import os
import io
import glob
import base64
import sys
import importlib
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torchvision.transforms as T
import pickle
from contextlib import asynccontextmanager
from typing import List

# ==========================================
# 导入本地模块
# ==========================================
from config.database import engine, Base
from api import auth_router
from metrics_calc import calculate_metrics_from_masks

# 必须显式导入数据库模型，SQLAlchemy 才能识别并建表
import db_models.user_models

# 导入核心分割模型
from our_model.JointOCTAMamba import JointOCTAMamba

CLASS_PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "class"))
if CLASS_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, CLASS_PACKAGE_DIR)

class_model_module = importlib.import_module("src.model")
build_dualbranch_model = class_model_module.build_model

# ==========================================
# 全局变量与配置
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
APP_TITLE = "Multi-Agent OCTA API (MySQL + Redis + LangGraph + Vision)"

MODEL_3M = None
MODEL_6M = None

ENSEMBLE_3M = []
ENSEMBLE_6M = []
CLASSIFY_CLASS_NAMES_3M = []
CLASSIFY_CLASS_NAMES_6M = []
CLASSIFY_MODE_3M = "img+tabletext"
CLASSIFY_MODE_6M = "img+tabletext"
CLASSIFY_RUNTIME_3M = {}
CLASSIFY_RUNTIME_6M = {}

DISEASE_CN = {
    "AMD": "老年性黄斑变性 (AMD)",
    "CNV": "脉络膜新生血管 (CNV)",
    "CSC": "中央浆液性脉络膜视网膜病变 (CSC)",
    "DR": "糖尿病视网膜病变 (DR)",
    "NORMAL": "正常 (NORMAL)",
    "OTHERS": "其他 (OTHERS)",
    "RVO": "视网膜静脉阻塞 (RVO)",
}

CLASS_LABEL_ALIASES = {
    "OTHER": "OTHERS",
    "OTHERS": "OTHERS",
    "NORMAL": "NORMAL",
    "AMD": "AMD",
    "DR": "DR",
    "CNV": "CNV",
    "CSC": "CSC",
    "RVO": "RVO",
}
EXPECTED_CLASS_NAMES_3M = ["NORMAL", "AMD", "DR", "CNV"]
EXPECTED_CLASS_NAMES_6M = ["NORMAL", "AMD", "DR", "CNV", "CSC", "RVO", "OTHERS"]

class ClassificationRequest(BaseModel):
    scan_type: str
    original_image_base64: str
    rv_mask_base64: str
    faz_mask_base64: str
    metrics: dict


CLASSIFY_METRIC_KEYS = [
    "faz_area_px",
    "faz_perim_px",
    "faz_circularity",
    "rv_density",
    "rv_flow_area_px",
    "rv_line_density_px-1",
    "rv_branch_points",
    "faz300_sim_density",
]

CLASSIFY_CFG_3M = {
    "model_type": "dual_timm",
    "image_backbone_name": "resnet50",
    "mask_backbone_name": "resnet18",
    "pretrained": True,
    "freeze_image_backbone": False,
    "freeze_mask_backbone": False,
    "img_size": 224,
    "sample_mode": "image_mask_metrics",
    "mask_size": 224,
    "image_cols": ["orig_path"],
    "num_classes": 4,
    "metrics_cols": CLASSIFY_METRIC_KEYS,
    "metrics_hidden_dim": 128,
    "image_feat_dim": 512,
    "mask_feat_dim": 256,
    "dropout": 0.2,
}

CLASSIFY_CFG_6M = {
    "model_type": "dual_timm",
    "image_backbone_name": "resnet50",
    "mask_backbone_name": "resnet18",
    "pretrained": True,
    "freeze_image_backbone": False,
    "freeze_mask_backbone": True,
    "img_size": 224,
    "sample_mode": "image_mask_metrics",
    "mask_size": 224,
    "aux_mode": "fusion",
    "image_cols": ["orig_path"],
    "num_classes": 7,
    "metrics_cols": CLASSIFY_METRIC_KEYS,
    "metrics_hidden_dim": 128,
    "image_feat_dim": 512,
    "mask_feat_dim": 32,
    "dropout": 0.3,
}

# =======================================================================
# SafePickle
# =======================================================================
class NumpySafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

class SafePickleModule:
    Unpickler = NumpySafeUnpickler

    @staticmethod
    def load(file, **kwargs):
        return NumpySafeUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(b, **kwargs):
        return NumpySafeUnpickler(io.BytesIO(b), **kwargs).load()

# ==========================================
# 模型加载辅助函数
# ==========================================
def load_checkpoint_into_model(model, ckpt: dict):
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return
    raise RuntimeError(f"Unsupported checkpoint format. Keys={list(ckpt.keys())}")


def load_dualbranch_classifier(cfg: dict, ckpt_path: str, device: torch.device) -> dict:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Classification checkpoint not found: {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
        pickle_module=SafePickleModule,
    )
    runtime_cfg = dict(cfg)
    runtime_cfg["num_classes"] = int(ckpt.get("num_classes", runtime_cfg["num_classes"]))

    model = build_dualbranch_model(runtime_cfg)
    load_checkpoint_into_model(model, ckpt)
    model.to(device)
    model.eval()

    metrics_mean = ckpt.get("metrics_mean")
    metrics_std = ckpt.get("metrics_std")
    if metrics_mean is not None:
        metrics_mean = np.array(metrics_mean, dtype=np.float32)
    if metrics_std is not None:
        metrics_std = np.array(metrics_std, dtype=np.float32)

    raw_label_names = ckpt.get("label_names", [])
    label_names = [CLASS_LABEL_ALIASES.get(str(name).upper().strip(), str(name).upper().strip()) for name in raw_label_names]

    return {
        "model": model,
        "ckpt_path": ckpt_path,
        "cfg": runtime_cfg,
        "label_names": label_names,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "metrics_cols": list(runtime_cfg.get("metrics_cols", [])),
        "img_size": int(runtime_cfg.get("img_size", 224)),
        "mask_size": int(runtime_cfg.get("mask_size", 224)),
    }

# ==========================================
# 统一生命周期管理器 (数据库 + AI模型)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 数据库建表
    print("⏳ 正在初始化数据库表...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 数据库表初始化完成！")
    except Exception as e:
        print(f"❌ 数据库表初始化失败: {e}")

    # 2. AI 模型加载
    global MODEL_3M, MODEL_6M
    global CLASSIFY_CLASS_NAMES_3M, CLASSIFY_MODE_3M, CLASSIFY_RUNTIME_3M
    global CLASSIFY_CLASS_NAMES_6M, CLASSIFY_MODE_6M, CLASSIFY_RUNTIME_6M

    print("\n>>> [1/2] Loading Vision Agent (Segmentation) Models...")
    try:
        MODEL_3M = JointOCTAMamba(tasks=["OCTA500_3M"], faz_crop_size=224, end_to_end=False)
        MODEL_3M.load_state_dict(torch.load("./pth/3m.pth", map_location=DEVICE, pickle_module=SafePickleModule))
        MODEL_3M.to(DEVICE)
        MODEL_3M.eval()

        MODEL_6M = JointOCTAMamba(tasks=["OCTA500_6M"], faz_crop_size=224, end_to_end=False)
        MODEL_6M.load_state_dict(torch.load("./pth/6m.pth", map_location=DEVICE, pickle_module=SafePickleModule))
        MODEL_6M.to(DEVICE)
        MODEL_6M.eval()

        print(f"✅ Both 3M and 6M segmentation models loaded on {DEVICE}")
    except Exception as e:
        print(f"❌ Failed to load segmentation models: {e}")

    print("\n>>> [2/2] Loading Classification Agent Models...")
    try:
        CLASSIFY_RUNTIME_3M = load_dualbranch_classifier(
            CLASSIFY_CFG_3M,
            "./class/dualbranch_resnet_3mm/last.pt",
            DEVICE,
        )
        CLASSIFY_CLASS_NAMES_3M = list(CLASSIFY_RUNTIME_3M.get("label_names", []))
        if CLASSIFY_CLASS_NAMES_3M != EXPECTED_CLASS_NAMES_3M:
            raise RuntimeError(f"3M classifier label_names mismatch: {CLASSIFY_CLASS_NAMES_3M}")
        CLASSIFY_MODE_3M = "image_mask_metrics"
        print("✅ Loaded 3M classifier from ./class/dualbranch_resnet_3mm/last.pt")
    except Exception as e:
        CLASSIFY_RUNTIME_3M = {}
        print(f"❌ Failed to load 3M classifier: {e}")

    try:
        CLASSIFY_RUNTIME_6M = load_dualbranch_classifier(
            CLASSIFY_CFG_6M,
            "./class/dualbranch_resnet_6mm_transformer_seed_42/last.pt",
            DEVICE,
        )
        CLASSIFY_CLASS_NAMES_6M = list(CLASSIFY_RUNTIME_6M.get("label_names", []))
        if CLASSIFY_CLASS_NAMES_6M != EXPECTED_CLASS_NAMES_6M:
            raise RuntimeError(f"6M classifier label_names mismatch: {CLASSIFY_CLASS_NAMES_6M}")
        CLASSIFY_MODE_6M = "image_mask_metrics"
        print("✅ Loaded 6M classifier from ./class/dualbranch_resnet_6mm_transformer_seed_42/last.pt")
    except Exception as e:
        CLASSIFY_RUNTIME_6M = {}
        print(f"❌ Failed to load 6M classifier: {e}")
    print("✅ System Startup Complete.\n")

    # 挂起运行
    yield 

    # 3. 关闭清理
    print("🛑 服务正在关闭...")

# ==========================================
# FastAPI 实例与路由注册 (全局唯一)
# ==========================================
app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载鉴权路由 (里面会自动使用数据库)
app.include_router(auth_router.router)
# app.include_router(chat_router.router)  # 之后写好分离的 chat 路由可以取消注释

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": APP_TITLE}

# ==========================================
# 图像处理辅助函数
# ==========================================
def mask_to_base64(mask_array: np.ndarray) -> str:
    mask_uint8 = (mask_array.astype(np.uint8) * 255)
    pil_img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def decode_mask_to_tensor(b64_str: str) -> torch.Tensor:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data)).convert("L")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.NEAREST)
    u8 = np.array(img)
    mask = (u8 > 0).astype(np.float32)
    return torch.from_numpy(mask)


def decode_base64_image(b64_str: str, mode: str = "RGB") -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))
    if mode:
        img = img.convert(mode)
    return img


def decode_mask_to_pil(b64_str: str, size=(224, 224)) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data)).convert("L")
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return img


def build_mask_fusion_pil(
    original_image: Image.Image,
    rv_mask: Image.Image,
    faz_mask: Image.Image,
    out_size: int = 224,
) -> Image.Image:
    orig = original_image.convert("L").resize((out_size, out_size), Image.BILINEAR)
    rv = rv_mask.convert("L").resize((out_size, out_size), Image.NEAREST)
    faz = faz_mask.convert("L").resize((out_size, out_size), Image.NEAREST)

    orig_np = np.array(orig, dtype=np.uint8)
    rv_bin = np.where(np.array(rv, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    faz_bin = np.where(np.array(faz, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    rgb = np.stack([orig_np, rv_bin, faz_bin], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def normalize_metrics(values: List[float], runtime: dict) -> np.ndarray:
    arr = np.array(values, dtype=np.float32)
    mean = runtime.get("metrics_mean")
    std = runtime.get("metrics_std")
    if mean is not None and std is not None and len(mean) == len(arr) and len(std) == len(arr):
        safe_std = np.where(std == 0, 1.0, std)
        arr = (arr - mean) / safe_std
    return arr.astype(np.float32)

def resize_binary_mask(mask_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    mask_uint8 = (mask_array.astype(np.uint8) * 255)
    pil_img = Image.fromarray(mask_uint8, mode="L")
    pil_img = pil_img.resize(target_size, Image.NEAREST)
    resized = np.array(pil_img)
    return (resized > 0)

def to_prob_map(pred_tensor: torch.Tensor) -> torch.Tensor:
    t_min = float(pred_tensor.detach().amin().item())
    t_max = float(pred_tensor.detach().amax().item())
    if t_min < 0.0 or t_max > 1.0:
        return torch.sigmoid(pred_tensor)
    return pred_tensor

# ==========================================
# Agent 1: 视觉分割智能体接口
# ==========================================
@app.post("/api/v1/agent/vision/analyze")
async def analyze_octa_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    width, height = pil_image.size
    scan_type = "Unknown"
    active_model = None

    if 280 <= width <= 320:
        scan_type = "3M"
        active_model = MODEL_3M
    elif 380 <= width <= 420:
        scan_type = "6M"
        active_model = MODEL_6M
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported image size: {width}x{height}.")

    if active_model is None:
        raise HTTPException(status_code=500, detail="Target model is not initialized.")

    input_tensor = T.ToTensor()(pil_image).unsqueeze(0).to(DEVICE)

    try:
        with torch.no_grad():
            pred_dict = active_model(input_tensor)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation model forward failed on original-size input ({width}x{height}): {str(e)}"
        )

    pred_rv_tensor = None
    pred_faz_tensor = None

    for task_name, pred_tensor in pred_dict.items():
        task_name_low = task_name.lower()
        if "rv" in task_name_low:
            pred_rv_tensor = pred_tensor
        elif "faz" in task_name_low:
            pred_faz_tensor = pred_tensor

    if pred_rv_tensor is None or pred_faz_tensor is None:
        raise HTTPException(status_code=500, detail="Model output missing 'rv' or 'faz' segmentation.")

    pred_rv_prob = to_prob_map(pred_rv_tensor)
    pred_faz_prob = to_prob_map(pred_faz_tensor)

    pred_rv_mask_raw = (pred_rv_prob > 0.5).squeeze().detach().cpu().numpy().astype(bool)
    pred_faz_mask_raw = (pred_faz_prob > 0.5).squeeze().detach().cpu().numpy().astype(bool)

    pred_rv_mask = resize_binary_mask(pred_rv_mask_raw, target_size=(224, 224))
    pred_faz_mask = resize_binary_mask(pred_faz_mask_raw, target_size=(224, 224))

    rv_density_check = pred_rv_mask.sum() / pred_rv_mask.size
    is_valid_octa = True
    if rv_density_check < 0.01 or rv_density_check > 0.80:
        is_valid_octa = False

    metrics = calculate_metrics_from_masks(pred_faz_mask, pred_rv_mask)

    rv_b64 = mask_to_base64(pred_rv_mask)
    faz_b64 = mask_to_base64(pred_faz_mask)

    raw_h, raw_w = pred_rv_mask_raw.shape[-2], pred_rv_mask_raw.shape[-1]

    return {
        "status": "success",
        "agent": "Vision_JointOCTAMamba_Agent",
        "image_metadata": {
            "original_size": [width, height],
            "model_output_size_before_resize": [raw_w, raw_h],
            "final_mask_size": [224, 224],
            "scan_type": scan_type,
            "is_valid_octa": is_valid_octa,
            "validation_msg": "Valid OCTA" if is_valid_octa else "Warning: Not a valid OCTA."
        },
        "metrics": metrics,
        "visualizations": {
            "rv_mask_base64": f"data:image/png;base64,{rv_b64}",
            "faz_mask_base64": f"data:image/png;base64,{faz_b64}"
        }
    }

# ==========================================
# Agent 2: 多模态分类智能体接口
# ==========================================
@app.post("/api/v1/agent/classify")
async def classify_disease(req: ClassificationRequest):
    scan_type = str(req.scan_type or "").upper().strip()
    if scan_type.startswith("3"):
        runtime = CLASSIFY_RUNTIME_3M
    elif scan_type.startswith("6"):
        runtime = CLASSIFY_RUNTIME_6M
    else:
        raise HTTPException(status_code=400, detail="Invalid scan_type. Must be 3M or 6M.")

    if not runtime or runtime.get("model") is None:
        raise HTTPException(status_code=500, detail=f"No classification model loaded for {scan_type}.")

    try:
        original_image = decode_base64_image(req.original_image_base64, mode="RGB")
        rv_mask = decode_mask_to_pil(req.rv_mask_base64, size=(224, 224))
        faz_mask = decode_mask_to_pil(req.faz_mask_base64, size=(224, 224))
        aux_image = build_mask_fusion_pil(
            original_image=original_image,
            rv_mask=rv_mask,
            faz_mask=faz_mask,
            out_size=int(runtime.get("mask_size", 224)),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode classification inputs: {str(e)}")

    metric_values = [float(req.metrics.get(k, 0.0)) for k in runtime.get("metrics_cols", CLASSIFY_METRIC_KEYS)]
    normalized_metrics = normalize_metrics(metric_values, runtime)
    metrics_tensor = torch.tensor([normalized_metrics], dtype=torch.float32)

    model = runtime["model"]
    class_names = list(runtime.get("label_names", []))
    if not class_names:
        raise HTTPException(status_code=500, detail=f"Missing label_names for {scan_type} classifier.")

    with torch.no_grad():
        output = model(
            images=[original_image],
            image_counts=[1],
            aux_images=[aux_image],
            aux_image_counts=[1],
            metrics=metrics_tensor,
        )
        logits = output["logits"]
        probs = F.softmax(logits.float(), dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label_en = CLASS_LABEL_ALIASES.get(str(class_names[pred_idx]).upper().strip(), str(class_names[pred_idx]).upper().strip())
    pred_label_cn = DISEASE_CN.get(pred_label_en, pred_label_en)
    confidence = float(probs[pred_idx])
    distribution = {
        CLASS_LABEL_ALIASES.get(str(class_names[i]).upper().strip(), str(class_names[i]).upper().strip()): round(float(probs[i]), 4)
        for i in range(len(class_names))
    }

    return {
        "status": "success",
        "agent": "Classification_DualBranch_Agent",
        "prediction": {
            "label_en": pred_label_en,
            "label_cn": pred_label_cn,
            "confidence": round(confidence, 4),
            "distribution": distribution
        },
        "meta": {
            "checkpoint": runtime.get("ckpt_path"),
            "mode_used": "image_mask_metrics",
            "metrics_keys": runtime.get("metrics_cols", []),
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
