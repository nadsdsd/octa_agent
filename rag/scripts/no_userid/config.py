# import os
# from functools import lru_cache

# import redis
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# try:
#     from huggingface_hub import snapshot_download
# except Exception:
#     snapshot_download = None


# # =========================
# # Helpers
# # =========================
# def parse_bool(value: str | None, default: bool = False) -> bool:
#     if value is None:
#         return default
#     return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# def parse_csv_set(value: str | None) -> set[str]:
#     if not value:
#         return set()
#     return {item.strip() for item in value.split(",") if item.strip()}


# # =========================
# # Environment / Constants
# # =========================
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# os.environ.setdefault("HF_HOME", os.path.abspath(os.getenv("HF_HOME", "./hf_cache")))

# APP_TITLE = "OCTA Agent Router API (LangGraph + Redis + Async + RAG)"
# TOOL_BACKEND_URL = os.getenv("TOOL_BACKEND_URL", "http://127.0.0.1:8000")
# HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
# REQUEST_LOCK_TTL_SECONDS = int(os.getenv("REQUEST_LOCK_TTL_SECONDS", "180"))

# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# REDIS_DB = int(os.getenv("REDIS_DB", "0"))
# SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
# DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "test_doctor_001")
# UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
# CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# API_ENDPOINTS = {
#     "vision_analyze": "/api/v1/agent/vision/analyze",
#     "diagnosis_classify": "/api/v1/agent/classify",
# }

# DEFAULT_ENABLED_AGENTS = {
#     "intent_node": True,
#     "vision_node": True,
#     "diagnosis_node": True,
#     "physical_node": True,
#     "pathology_node": True,
#     "rag_node": True,
#     "final_node": True,
# }
# DISABLED_AGENTS = parse_csv_set(os.getenv("DISABLED_AGENTS"))

# # =========================
# # LLM Providers
# # =========================
# DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "SILICONFLOW").upper()
# PHYSICAL_LLM_PROVIDER = os.getenv("PHYSICAL_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).upper()
# PATHOLOGY_LLM_PROVIDER = os.getenv("PATHOLOGY_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).upper()

# # ZHIPU
# GLM_MODEL = os.getenv("GLM_MODEL", "glm-4.7-flash")
# ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")

# # OLLAMA
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.22.96.1:11434/v1")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# # SILICONFLOW (OpenAI Compatible)
# SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
# SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
# # 这里默认给一个示例模型；如果你控制台显示的免费模型 ID 不同，直接改环境变量即可
# SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3.5-4B")

# LLM_PROVIDERS = {
#     "OLLAMA": {
#         "type": "openai_compatible",
#         "base_url": OLLAMA_BASE_URL,
#         "model": OLLAMA_MODEL,
#         "api_key": "ollama",
#         "enabled": True,
#     },
#     "ZHIPU": {
#         "type": "zhipu",
#         "model": GLM_MODEL,
#         "api_key": ZHIPUAI_API_KEY,
#         "enabled": bool(ZHIPUAI_API_KEY),
#     },
#     "SILICONFLOW": {
#         "type": "openai_compatible",
#         "base_url": SILICONFLOW_BASE_URL,
#         "model": SILICONFLOW_MODEL,
#         "api_key": SILICONFLOW_API_KEY,
#         "enabled": bool(SILICONFLOW_API_KEY),
#     },
# }

# LLM_FALLBACKS = {
#     "SILICONFLOW": ["ZHIPU", "OLLAMA"],
#     "ZHIPU": ["SILICONFLOW", "OLLAMA"],
#     "OLLAMA": ["SILICONFLOW", "ZHIPU"],
# }

# # =========================
# # Embeddings / Vector DB
# # =========================
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
# EMBEDDING_MODEL_DIR = os.path.abspath(os.getenv("EMBEDDING_MODEL_DIR", "./models/bge-small-zh-v1.5"))
# EMBEDDING_LOCAL_ONLY = parse_bool(os.getenv("EMBEDDING_LOCAL_ONLY"), True)
# EMBEDDING_AUTO_DOWNLOAD = parse_bool(os.getenv("EMBEDDING_AUTO_DOWNLOAD"), False)

# # =========================
# # Prompts / Domain data
# # =========================
# SYSTEM_PROMPT = (
#     "你是一个专业的眼底影像科AI助手。你的职责是基于工作流结果，"
#     "用专业、温和、清晰的中文生成最终答复。"
#     "\n要求："
#     "\n1. 若系统拦截或图像无效，先解释原因，再说明为何停止后续诊断。"
#     "\n2. 若用户只要求分割/指标，禁止输出疾病诊断。"
#     "\n3. 若已有诊断但用户未明确索要治疗方案，不要主动展开治疗建议。"
#     "\n4. 只有在用户明确要求方案/治疗/用药/报告/医学咨询时，才结合RAG内容生成建议。"
#     "\n5. 严禁输出代码块；不夸大结论；避免替代线下医生最终诊断。"
# )

# DISEASE_PHYSICAL_STATS = """
# AMD (n = 43)
# faz_area_mm2: 0.4613 ± 0.1575
# faz_perim_mm: 3.2674 ± 0.7180
# faz_circularity: 0.5498 ± 0.1182
# rv_density: 0.0854 ± 0.0117
# rv_flow_area_mm2: 3.0755 ± 0.4207
# rv_line_density_mm-1: 2.5424 ± 0.3472
# rv_branch_points: 2226.0465 ± 375.3366
# faz300_sim_density: 0.0209 ± 0.0114

# CNV (n = 11)
# faz_area_mm2: 0.3706 ± 0.0961
# faz_perim_mm: 2.8141 ± 0.4813
# faz_circularity: 0.5911 ± 0.0887
# rv_density: 0.0959 ± 0.0135
# rv_flow_area_mm2: 3.4531 ± 0.4844
# rv_line_density_mm-1: 2.7552 ± 0.2837
# rv_branch_points: 2423.3636 ± 271.6127
# faz300_sim_density: 0.0189 ± 0.0082

# CSC (n = 14)
# faz_area_mm2: 0.3464 ± 0.1193
# faz_perim_mm: 2.7180 ± 0.5404
# faz_circularity: 0.5815 ± 0.0926
# rv_density: 0.0904 ± 0.0088
# rv_flow_area_mm2: 3.2551 ± 0.3162
# rv_line_density_mm-1: 2.6432 ± 0.2240
# rv_branch_points: 2293.7857 ± 236.9545
# faz300_sim_density: 0.0162 ± 0.0071

# DR (n = 35)
# faz_area_mm2: 0.4428 ± 0.1327
# faz_perim_mm: 3.2273 ± 0.6971
# faz_circularity: 0.5431 ± 0.1067
# rv_density: 0.0977 ± 0.0174
# rv_flow_area_mm2: 3.5158 ± 0.6281
# rv_line_density_mm-1: 2.8149 ± 0.4592
# rv_branch_points: 2428.9429 ± 420.5443
# faz300_sim_density: 0.0241 ± 0.0131

# NORMAL (n = 91)
# faz_area_mm2: 0.3375 ± 0.1063
# faz_perim_mm: 2.5905 ± 0.4774
# faz_circularity: 0.6296 ± 0.0988
# rv_density: 0.0944 ± 0.0128
# rv_flow_area_mm2: 3.3999 ± 0.4600
# rv_line_density_mm-1: 2.7328 ± 0.2875
# rv_branch_points: 2395.9341 ± 286.6391
# faz300_sim_density: 0.0209 ± 0.0088

# RVO (n = 10)
# faz_area_mm2: 0.3749 ± 0.0962
# faz_perim_mm: 3.0970 ± 0.5104
# faz_circularity: 0.5003 ± 0.1216
# rv_density: 0.1031 ± 0.0126
# rv_flow_area_mm2: 3.7119 ± 0.4551
# rv_line_density_mm-1: 2.9381 ± 0.3099
# rv_branch_points: 2627.1000 ± 332.2002
# faz300_sim_density: 0.0280 ± 0.0093
# """

# DISEASE_SYMPTOMS = """
# 1. AMD: 中心视力模糊、视物变形（直线弯曲）、颜色敏感度下降、暗适应差、中心暗点。
# 2. CNV: 突发性视力断崖式下降、急剧的视物变形和变小、突然出现的眼前黑影、眼前闪光感。
# 3. CSC: 青年压力大男性多发。单眼突然视力模糊（像隔着水雾）、视物变暗发黄、视物变小变远、相对性暗点。
# 4. DR: 糖尿病史。早期无症状，后期视力波动不稳定、飞蚊症突然加重（新生血管破裂出血）、夜间视力变差。
# 5. RVO: 多见高血压/动脉硬化。早晨醒来无痛性突发视力下降、视野局部缺损被黑影遮挡。
# """

# TRANSIENT_CONTEXT_KEYS = [
#     "rv_mask_base64",
#     "faz_mask_base64",
#     "metrics",
#     "scan_type",
#     "diagnosis_result",
#     "physical_metrics",
#     "image_diagnosis_label",
#     "physical_diagnosis_label",
#     "pathology_diagnosis_result",
#     "last_disease_label",
#     "symptoms_requested",
# ]

# FOLLOWUP_SAFE_KEYS = [
#     "rv_mask_base64",
#     "faz_mask_base64",
#     "metrics",
#     "scan_type",
#     "diagnosis_result",
#     "physical_metrics",
#     "image_diagnosis_label",
#     "physical_diagnosis_label",
#     "pathology_diagnosis_result",
#     "last_disease_label",
#     "symptoms_requested",
# ]


# def build_enabled_agents() -> dict[str, bool]:
#     enabled = dict(DEFAULT_ENABLED_AGENTS)
#     for node_name in DISABLED_AGENTS:
#         if node_name in enabled:
#             enabled[node_name] = False
#     return enabled


# @lru_cache(maxsize=1)
# def get_redis_client() -> redis.Redis:
#     client = redis.Redis(
#         host=REDIS_HOST,
#         port=REDIS_PORT,
#         db=REDIS_DB,
#         decode_responses=True,
#     )
#     client.ping()
#     return client


# def ensure_local_embedding_model() -> str:
#     os.makedirs(os.path.dirname(EMBEDDING_MODEL_DIR), exist_ok=True)
#     if os.path.isdir(EMBEDDING_MODEL_DIR) and os.listdir(EMBEDDING_MODEL_DIR):
#         return EMBEDDING_MODEL_DIR

#     if EMBEDDING_LOCAL_ONLY and not EMBEDDING_AUTO_DOWNLOAD:
#         raise RuntimeError(
#             "本地 embedding 模型目录为空。请先执行 prepare_models.py 下载模型，"
#             f"或将 EMBEDDING_AUTO_DOWNLOAD=true。目标目录: {EMBEDDING_MODEL_DIR}"
#         )

#     if not EMBEDDING_AUTO_DOWNLOAD:
#         return EMBEDDING_MODEL_NAME

#     if snapshot_download is None:
#         raise RuntimeError("当前环境未安装 huggingface_hub，无法自动下载 embedding 模型。")

#     snapshot_download(
#         repo_id=EMBEDDING_MODEL_NAME,
#         local_dir=EMBEDDING_MODEL_DIR,
#         local_dir_use_symlinks=False,
#         resume_download=True,
#     )
#     return EMBEDDING_MODEL_DIR


# @lru_cache(maxsize=1)
# def get_embeddings() -> HuggingFaceEmbeddings:
#     model_ref = ensure_local_embedding_model()
#     model_kwargs = {}
#     if os.path.isdir(model_ref):
#         model_kwargs["local_files_only"] = True
#     return HuggingFaceEmbeddings(model_name=model_ref, model_kwargs=model_kwargs)


# @lru_cache(maxsize=1)
# def get_vector_db() -> Chroma:
#     return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
import os
from functools import lru_cache

import redis
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


# =========================
# Helpers
# =========================
def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


# =========================
# Environment / Constants
# =========================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HOME", os.path.abspath(os.getenv("HF_HOME", "./hf_cache")))

APP_TITLE = "OCTA Agent Router API (LangGraph + Redis + Async + RAG)"
TOOL_BACKEND_URL = os.getenv("TOOL_BACKEND_URL", "http://127.0.0.1:8000")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
REQUEST_LOCK_TTL_SECONDS = int(os.getenv("REQUEST_LOCK_TTL_SECONDS", "180"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "test_doctor_001")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# =========================
# Debug / Trace
# =========================
DEBUG_TRACE_ENABLED = parse_bool(os.getenv("DEBUG_TRACE_ENABLED"), True)
TRACE_RESPONSE_PREVIEW_CHARS = int(os.getenv("TRACE_RESPONSE_PREVIEW_CHARS", "400"))

# =========================
# Probe / Ranking
# =========================
PROBE_TIMEOUT_SECONDS = float(os.getenv("PROBE_TIMEOUT_SECONDS", "2.0"))
PROBE_CACHE_TTL_SECONDS = int(os.getenv("PROBE_CACHE_TTL_SECONDS", "60"))

# =========================
# Fixed API paths
# =========================
API_ENDPOINTS = {
    "vision_analyze": "/api/v1/agent/vision/analyze",
    "diagnosis_classify": "/api/v1/agent/classify",
}

# =========================
# Multiple service backends
# 如果只填一个地址，也可以正常工作
# 如果填多个地址，utils.py 里可以按测速动态选择
# =========================
VISION_BACKEND_URLS = parse_csv_list(os.getenv("VISION_BACKEND_URLS", TOOL_BACKEND_URL)) or [TOOL_BACKEND_URL]
DIAGNOSIS_BACKEND_URLS = parse_csv_list(os.getenv("DIAGNOSIS_BACKEND_URLS", TOOL_BACKEND_URL)) or [TOOL_BACKEND_URL]

SERVICE_CANDIDATES = {
    "vision_analyze": [
        {
            "name": f"vision_{idx}",
            "base_url": base_url.rstrip("/"),
            "path": API_ENDPOINTS["vision_analyze"],
            "probe_url": f"{base_url.rstrip('/')}/health",
        }
        for idx, base_url in enumerate(VISION_BACKEND_URLS, start=1)
    ],
    "diagnosis_classify": [
        {
            "name": f"diagnosis_{idx}",
            "base_url": base_url.rstrip("/"),
            "path": API_ENDPOINTS["diagnosis_classify"],
            "probe_url": f"{base_url.rstrip('/')}/health",
        }
        for idx, base_url in enumerate(DIAGNOSIS_BACKEND_URLS, start=1)
    ],
}

# =========================
# Agent enable / ablation
# =========================
DEFAULT_ENABLED_AGENTS = {
    "intent_node": True,
    "vision_node": True,
    "diagnosis_node": True,
    "physical_node": True,
    "pathology_node": True,
    "rag_node": True,
    "final_node": True,
}
DISABLED_AGENTS = parse_csv_set(os.getenv("DISABLED_AGENTS"))

# =========================
# LLM Providers
# 默认改成 SiliconFlow
# =========================
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "SILICONFLOW").upper()
PHYSICAL_LLM_PROVIDER = os.getenv("PHYSICAL_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).upper()
PATHOLOGY_LLM_PROVIDER = os.getenv("PATHOLOGY_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).upper()

# ZHIPU
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4.7-flash")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M")

# OLLAMA
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.22.96.1:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# SILICONFLOW (OpenAI Compatible)
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-yplegeddsyscvtqfnwqwdiucgjvnocceazrqvbzanoupgkvi")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3.5-4B")

LLM_PROVIDERS = {
    "OLLAMA": {
        "type": "openai_compatible",
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "api_key": "ollama",
        "probe_url": f"{OLLAMA_BASE_URL.rstrip('/')}/models",
        "enabled": True,
    },
    "ZHIPU": {
        "type": "zhipu",
        "model": GLM_MODEL,
        "api_key": ZHIPUAI_API_KEY,
        "probe_url": "",
        "enabled": bool(ZHIPUAI_API_KEY),
    },
    "SILICONFLOW": {
        "type": "openai_compatible",
        "base_url": SILICONFLOW_BASE_URL,
        "model": SILICONFLOW_MODEL,
        "api_key": SILICONFLOW_API_KEY,
        "probe_url": f"{SILICONFLOW_BASE_URL.rstrip('/')}/models",
        "enabled": bool(SILICONFLOW_API_KEY),
    },
}

LLM_FALLBACKS = {
    "SILICONFLOW": ["ZHIPU", "OLLAMA"],
    "ZHIPU": ["SILICONFLOW", "OLLAMA"],
    "OLLAMA": ["SILICONFLOW", "ZHIPU"],
}

# =========================
# Embeddings / Vector DB
# =========================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
EMBEDDING_MODEL_DIR = os.path.abspath(os.getenv("EMBEDDING_MODEL_DIR", "./models/bge-small-zh-v1.5"))
EMBEDDING_LOCAL_ONLY = parse_bool(os.getenv("EMBEDDING_LOCAL_ONLY"), True)
EMBEDDING_AUTO_DOWNLOAD = parse_bool(os.getenv("EMBEDDING_AUTO_DOWNLOAD"), False)

# =========================
# Prompts / Domain data
# =========================
SYSTEM_PROMPT = (
    "你是一个专业的眼底影像科AI助手。你的职责是基于工作流结果，"
    "用专业、温和、清晰的中文生成最终答复。"
    "\n要求："
    "\n1. 若系统拦截或图像无效，先解释原因，再说明为何停止后续诊断。"
    "\n2. 若用户只要求分割/指标，禁止输出疾病诊断。"
    "\n3. 若已有诊断但用户未明确索要治疗方案，不要主动展开治疗建议。"
    "\n4. 只有在用户明确要求方案/治疗/用药/报告/医学咨询时，才结合RAG内容生成建议。"
    "\n5. 严禁输出代码块；不夸大结论；避免替代线下医生最终诊断。"
)

DISEASE_PHYSICAL_STATS = """
AMD (n = 43)
faz_area_mm2: 0.4613 ± 0.1575
faz_perim_mm: 3.2674 ± 0.7180
faz_circularity: 0.5498 ± 0.1182
rv_density: 0.0854 ± 0.0117
rv_flow_area_mm2: 3.0755 ± 0.4207
rv_line_density_mm-1: 2.5424 ± 0.3472
rv_branch_points: 2226.0465 ± 375.3366
faz300_sim_density: 0.0209 ± 0.0114

CNV (n = 11)
faz_area_mm2: 0.3706 ± 0.0961
faz_perim_mm: 2.8141 ± 0.4813
faz_circularity: 0.5911 ± 0.0887
rv_density: 0.0959 ± 0.0135
rv_flow_area_mm2: 3.4531 ± 0.4844
rv_line_density_mm-1: 2.7552 ± 0.2837
rv_branch_points: 2423.3636 ± 271.6127
faz300_sim_density: 0.0189 ± 0.0082

CSC (n = 14)
faz_area_mm2: 0.3464 ± 0.1193
faz_perim_mm: 2.7180 ± 0.5404
faz_circularity: 0.5815 ± 0.0926
rv_density: 0.0904 ± 0.0088
rv_flow_area_mm2: 3.2551 ± 0.3162
rv_line_density_mm-1: 2.6432 ± 0.2240
rv_branch_points: 2293.7857 ± 236.9545
faz300_sim_density: 0.0162 ± 0.0071

DR (n = 35)
faz_area_mm2: 0.4428 ± 0.1327
faz_perim_mm: 3.2273 ± 0.6971
faz_circularity: 0.5431 ± 0.1067
rv_density: 0.0977 ± 0.0174
rv_flow_area_mm2: 3.5158 ± 0.6281
rv_line_density_mm-1: 2.8149 ± 0.4592
rv_branch_points: 2428.9429 ± 420.5443
faz300_sim_density: 0.0241 ± 0.0131

NORMAL (n = 91)
faz_area_mm2: 0.3375 ± 0.1063
faz_perim_mm: 2.5905 ± 0.4774
faz_circularity: 0.6296 ± 0.0988
rv_density: 0.0944 ± 0.0128
rv_flow_area_mm2: 3.3999 ± 0.4600
rv_line_density_mm-1: 2.7328 ± 0.2875
rv_branch_points: 2395.9341 ± 286.6391
faz300_sim_density: 0.0209 ± 0.0088

RVO (n = 10)
faz_area_mm2: 0.3749 ± 0.0962
faz_perim_mm: 3.0970 ± 0.5104
faz_circularity: 0.5003 ± 0.1216
rv_density: 0.1031 ± 0.0126
rv_flow_area_mm2: 3.7119 ± 0.4551
rv_line_density_mm-1: 2.9381 ± 0.3099
rv_branch_points: 2627.1000 ± 332.2002
faz300_sim_density: 0.0280 ± 0.0093
"""

DISEASE_SYMPTOMS = """
1. AMD: 中心视力模糊、视物变形（直线弯曲）、颜色敏感度下降、暗适应差、中心暗点。
2. CNV: 突发性视力断崖式下降、急剧的视物变形和变小、突然出现的眼前黑影、眼前闪光感。
3. CSC: 青年压力大男性多发。单眼突然视力模糊（像隔着水雾）、视物变暗发黄、视物变小变远、相对性暗点。
4. DR: 糖尿病史。早期无症状，后期视力波动不稳定、飞蚊症突然加重（新生血管破裂出血）、夜间视力变差。
5. RVO: 多见高血压/动脉硬化。早晨醒来无痛性突发视力下降、视野局部缺损被黑影遮挡。
"""

TRANSIENT_CONTEXT_KEYS = [
    "rv_mask_base64",
    "faz_mask_base64",
    "metrics",
    "scan_type",
    "diagnosis_result",
    "physical_metrics",
    "image_diagnosis_label",
    "physical_diagnosis_label",
    "pathology_diagnosis_result",
    "last_disease_label",
    "symptoms_requested",
]

FOLLOWUP_SAFE_KEYS = [
    "rv_mask_base64",
    "faz_mask_base64",
    "metrics",
    "scan_type",
    "diagnosis_result",
    "physical_metrics",
    "image_diagnosis_label",
    "physical_diagnosis_label",
    "pathology_diagnosis_result",
    "last_disease_label",
    "symptoms_requested",
]


def build_enabled_agents() -> dict[str, bool]:
    enabled = dict(DEFAULT_ENABLED_AGENTS)
    for node_name in DISABLED_AGENTS:
        if node_name in enabled:
            enabled[node_name] = False
    return enabled


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )
    client.ping()
    return client


def ensure_local_embedding_model() -> str:
    os.makedirs(os.path.dirname(EMBEDDING_MODEL_DIR), exist_ok=True)
    if os.path.isdir(EMBEDDING_MODEL_DIR) and os.listdir(EMBEDDING_MODEL_DIR):
        return EMBEDDING_MODEL_DIR

    if EMBEDDING_LOCAL_ONLY and not EMBEDDING_AUTO_DOWNLOAD:
        raise RuntimeError(
            "本地 embedding 模型目录为空。请先执行 prepare_models.py 下载模型，"
            f"或将 EMBEDDING_AUTO_DOWNLOAD=true。目标目录: {EMBEDDING_MODEL_DIR}"
        )

    if not EMBEDDING_AUTO_DOWNLOAD:
        return EMBEDDING_MODEL_NAME

    if snapshot_download is None:
        raise RuntimeError("当前环境未安装 huggingface_hub，无法自动下载 embedding 模型。")

    snapshot_download(
        repo_id=EMBEDDING_MODEL_NAME,
        local_dir=EMBEDDING_MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return EMBEDDING_MODEL_DIR


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    model_ref = ensure_local_embedding_model()
    model_kwargs = {}
    if os.path.isdir(model_ref):
        model_kwargs["local_files_only"] = True
    return HuggingFaceEmbeddings(model_name=model_ref, model_kwargs=model_kwargs)


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())