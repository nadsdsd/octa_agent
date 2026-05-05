import os
import re
import json
import base64
import mimetypes
import shutil
import asyncio
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict

import httpx
import redis
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from zhipuai import ZhipuAI

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================
# Environment / Constants
# =========================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

APP_TITLE = "OCTA Agent Router API (LangGraph + Redis + Async + RAG)"
TOOL_BACKEND_URL = os.getenv("TOOL_BACKEND_URL", "http://127.0.0.1:8000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "test_doctor_001")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp_uploads")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
GLM_ROUTER_MODEL = os.getenv("GLM_ROUTER_MODEL", "glm-4.7-flash")
GLM_REPORT_MODEL = os.getenv("GLM_REPORT_MODEL", "glm-4.7-flash")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M")
ZHIPUAI_API_KEY  = "9de2af25ee6a4ae8b8792c10df873e6d.Kobo0TQO8f2iTB4M"
if not ZHIPUAI_API_KEY:
    raise RuntimeError("请先设置环境变量 ZHIPUAI_API_KEY")

client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
)

print("⏳ 正在挂载本地医学知识库...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
print("✅ 本地知识库挂载完毕！")

try:
    redis_client.ping()
    print("✅ Redis 连接成功，当前使用手动 session_id 进行会话隔离。")
except Exception as exc:
    print(f"❌ Redis 连接失败，请检查服务是否启动: {exc}")
    raise


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


class SessionContext(TypedDict, total=False):
    rv_mask_base64: Optional[str]
    faz_mask_base64: Optional[str]
    metrics: Optional[Dict[str, Any]]
    scan_type: Optional[str]
    diagnosis_result: Optional[Dict[str, Any]]
    last_disease_label: Optional[str]


class AgentState(TypedDict, total=False):
    session_id: str
    user_text: str
    image_path: Optional[str]
    history: List[Dict[str, str]]
    context: SessionContext

    user_intent: Literal["segment_only", "diagnosis", "consult", "general_chat"]
    needs_vision: bool
    needs_diagnosis: bool
    needs_rag: bool
    route_reason: str

    vision_result: Optional[Dict[str, Any]]
    diagnosis_result: Optional[Dict[str, Any]]
    rag_result: Optional[str]
    rag_docs: Optional[List[str]]

    blocked: bool
    error: Optional[str]
    final_text: str
    progress_event: Optional[Dict[str, Any]]


# =========================
# Redis session helpers
# =========================
def get_session_state(session_id: str) -> Dict[str, Any]:
    raw = redis_client.get(f"agent_session:{session_id}")
    if raw:
        return json.loads(raw)
    return {
        "history": [{"role": "system", "content": SYSTEM_PROMPT}],
        "context": {
            "rv_mask_base64": None,
            "faz_mask_base64": None,
            "metrics": None,
            "scan_type": None,
            "diagnosis_result": None,
            "last_disease_label": None,
        },
    }


def save_session_state(session_id: str, state: Dict[str, Any]) -> None:
    redis_client.setex(
        f"agent_session:{session_id}",
        SESSION_TTL_SECONDS,
        json.dumps(state, ensure_ascii=False),
    )


# =========================
# Utility helpers
# =========================
def keep_recent_history(history: List[Dict[str, str]], max_turns: int = 15) -> List[Dict[str, str]]:
    if len(history) <= max_turns:
        return history
    return [history[0]] + history[-(max_turns - 1):]


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("空响应，无法解析 JSON")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def emit_stream_event(event: Optional[Dict[str, Any]]) -> None:
    if not event or get_stream_writer is None:
        return

    try:
        writer = get_stream_writer()
    except Exception:
        return

    if writer is None:
        return

    try:
        writer(event)
    except Exception:
        pass


def normalize_stream_chunk(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict) and "type" in chunk and "data" in chunk:
        return chunk

    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return {"type": chunk[0], "data": chunk[1]}

    if isinstance(chunk, dict):
        return {"type": "updates", "data": chunk}

    return {"type": "unknown", "data": chunk}


async def iter_graph_stream(initial_state: AgentState):
    stream_attempts = [
        {"stream_mode": ["custom", "updates"], "version": "v2"},
        {"stream_mode": "updates", "version": "v2"},
        {"stream_mode": "updates"},
    ]

    last_type_error: Optional[Exception] = None
    for kwargs in stream_attempts:
        try:
            async for raw_chunk in graph.astream(initial_state, **kwargs):
                yield normalize_stream_chunk(raw_chunk)
            return
        except TypeError as exc:
            last_type_error = exc
            continue

    if last_type_error:
        raise last_type_error


async def call_glm(messages: List[Dict[str, str]], model: str, temperature: float = 0.1) -> str:
    def _request() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    return await asyncio.to_thread(_request)


def heuristic_intent_fallback(user_text: str, has_image: bool, has_prior_dx: bool) -> Dict[str, Any]:
    text = (user_text or "").lower()

    segment_keywords = ["分割", "mask", "rv", "faz", "血流指标", "指标", "只看分割", "仅分割"]
    consult_keywords = ["治疗", "方案", "怎么治", "用药", "报告", "建议", "咨询", "指南", "医学知识"]
    diagnosis_keywords = ["诊断", "分析", "判断", "病变", "筛查", "看看这张图", "帮我看", "识别"]

    wants_segment = any(k in text for k in segment_keywords)
    wants_consult = any(k in text for k in consult_keywords)
    wants_diagnosis = any(k in text for k in diagnosis_keywords)

    if has_image and wants_segment and not wants_consult and not wants_diagnosis:
        return {
            "user_intent": "segment_only",
            "needs_vision": True,
            "needs_diagnosis": False,
            "needs_rag": False,
            "route_reason": "命中分割类关键词，且未要求诊断/方案。",
        }

    if wants_consult:
        return {
            "user_intent": "consult",
            "needs_vision": bool(has_image),
            "needs_diagnosis": bool(has_image and not has_prior_dx),
            "needs_rag": True,
            "route_reason": "命中治疗/方案/咨询类关键词。",
        }

    if has_image or wants_diagnosis:
        return {
            "user_intent": "diagnosis",
            "needs_vision": bool(has_image),
            "needs_diagnosis": bool(has_image or not has_prior_dx),
            "needs_rag": False,
            "route_reason": "默认进入图像分析/诊断路径。",
        }

    return {
        "user_intent": "general_chat",
        "needs_vision": False,
        "needs_diagnosis": False,
        "needs_rag": wants_consult,
        "route_reason": "无图像输入，按普通咨询或闲聊处理。",
    }


async def classify_intent_with_llm(state: AgentState) -> Dict[str, Any]:
    user_text = state.get("user_text", "")
    has_image = bool(state.get("image_path"))
    prior_label = state.get("context", {}).get("last_disease_label")
    has_prior_dx = bool(prior_label)

    router_prompt = f"""
你是医疗工作流路由器。请只输出一个 JSON 对象，不要输出任何额外说明。

任务：根据用户本轮输入、是否上传图像、是否已有历史诊断，决定是否需要走以下节点：
1. vision（图像分割/特征提取）
2. diagnosis（疾病诊断）
3. rag（医学知识检索）

规则：
- 如果用户只要求看分割、mask、FAZ/RV、指标，intent=segment_only，needs_vision=true，needs_diagnosis=false，needs_rag=false。
- 如果用户上传了图像并要求“分析/诊断/看看这张图”，intent=diagnosis，needs_vision=true，needs_diagnosis=true。
- 只有当用户明确要求治疗方案、用药建议、报告、医学咨询、指南知识时，needs_rag=true。
- 如果用户只是要诊断结果，没有明确要方案，needs_rag=false。
- 如果当前没有新图像，但历史中已经有诊断，而用户本轮只是在追问治疗/建议，则 needs_vision=false，needs_diagnosis=false，needs_rag=true。
- 如果没有图像且只是普通闲聊，intent=general_chat。

返回格式：
{{
  "user_intent": "segment_only | diagnosis | consult | general_chat",
  "needs_vision": true,
  "needs_diagnosis": false,
  "needs_rag": false,
  "route_reason": "一句中文解释"
}}

输入信息：
- 用户文本: {user_text or "<空>"}
- 是否上传图像: {str(has_image).lower()}
- 历史是否已有诊断: {str(has_prior_dx).lower()}
- 历史疾病标签: {prior_label or "<无>"}
    """.strip()

    content = await call_glm(
        messages=[{"role": "user", "content": router_prompt}],
        model=GLM_ROUTER_MODEL,
        temperature=0.0,
    )

    parsed = extract_json_object(content)
    validated = {
        "user_intent": parsed.get("user_intent", "diagnosis"),
        "needs_vision": bool(parsed.get("needs_vision", False)),
        "needs_diagnosis": bool(parsed.get("needs_diagnosis", False)),
        "needs_rag": bool(parsed.get("needs_rag", False)),
        "route_reason": str(parsed.get("route_reason", "LLM 已完成路由判断。")),
    }

    allowed = {"segment_only", "diagnosis", "consult", "general_chat"}
    if validated["user_intent"] not in allowed:
        raise ValueError(f"非法 intent: {validated['user_intent']}")

    return validated


def build_rag_query(state: AgentState) -> str:
    user_text = (state.get("user_text") or "").strip()
    disease_label = state.get("context", {}).get("last_disease_label") or ""
    if disease_label and user_text:
        return f"{disease_label} {user_text}"
    if disease_label:
        return f"{disease_label} 治疗方案 指南"
    if user_text:
        return user_text
    return "眼底疾病 治疗方案 指南"


def render_non_llm_final(state: AgentState) -> str:
    if state.get("error"):
        return f"本次流程未能完成：{state['error']}"

    if state.get("blocked"):
        return "图像已被系统判定为非标准 OCTA 或质量异常，因此本次停止后续诊断。建议重新上传符合要求的图像。"

    if state.get("user_intent") == "segment_only":
        metrics = state.get("context", {}).get("metrics")
        return f"已完成图像分割与指标提取。本次按你的要求只返回分割/指标信息，不做疾病诊断。当前指标：{metrics}"

    diagnosis = state.get("context", {}).get("last_disease_label")
    if diagnosis and state.get("needs_rag") and state.get("rag_result"):
        return f"初步判断为：{diagnosis}。结合知识库检索结果，已补充相关建议，请以线下专科医生面诊意见为准。"

    if diagnosis:
        return f"初步判断为：{diagnosis}。本轮未检索治疗方案，因为你没有明确要求方案/用药建议。"

    return "已完成本轮处理。"


# =========================
# LangGraph nodes
# =========================
async def intent_node(state: AgentState) -> Dict[str, Any]:
    has_image = bool(state.get("image_path"))
    has_prior_dx = bool(state.get("context", {}).get("last_disease_label"))

    progress = {"type": "thinking", "msg": "Node 1: 正在识别用户意图并规划工作流..."}
    emit_stream_event(progress)

    update: Dict[str, Any] = {
        "progress_event": progress
    }

    try:
        routing = await classify_intent_with_llm(state)
    except Exception:
        routing = heuristic_intent_fallback(state.get("user_text", ""), has_image, has_prior_dx)

    if routing["needs_vision"] and not has_image and not state.get("context", {}).get("metrics"):
        update.update(
            {
                **routing,
                "error": "当前没有可分析的图像，也没有历史视觉特征，无法执行分割/诊断。",
                "progress_event": {"type": "thinking", "msg": "Node 1: 已识别到缺少图像输入，准备直接结束。"},
            }
        )
        return update

    if routing["needs_diagnosis"] and not has_image and not state.get("context", {}).get("metrics"):
        update.update(
            {
                **routing,
                "error": "当前没有可用的视觉特征，无法执行疾病诊断。",
                "progress_event": {"type": "thinking", "msg": "Node 1: 缺少视觉特征，无法继续诊断。"},
            }
        )
        return update

    update.update(routing)
    return update


async def vision_node(state: AgentState) -> Dict[str, Any]:
    image_path = state.get("image_path")
    if not image_path:
        return {"error": "vision 节点被触发，但没有 image_path。"}

    progress = {"type": "thinking", "msg": "Node 2: 唤醒 Vision Agent，正在提取 OCTA 分割结果与血流指标..."}
    emit_stream_event(progress)

    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            with open(image_path, "rb") as file_obj:
                response = await http_client.post(
                    f"{TOOL_BACKEND_URL}/api/v1/agent/vision/analyze",
                    files={"file": file_obj},
                )

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "视觉服务调用失败")
            except Exception:
                detail = response.text
            return {
                "error": f"视觉服务返回异常：{detail}",
                "progress_event": progress,
            }

        data = response.json()
        image_meta = data.get("image_metadata", {})
        is_valid_octa = bool(image_meta.get("is_valid_octa", False))

        if not is_valid_octa:
            return {
                "blocked": True,
                "vision_result": data,
                "progress_event": progress,
                "error": "血管密度极度异常，系统判定为非标准 OCTA，已停止后续诊断。",
            }

        context = dict(state.get("context", {}))
        context.update(
            {
                "rv_mask_base64": data.get("visualizations", {}).get("rv_mask_base64"),
                "faz_mask_base64": data.get("visualizations", {}).get("faz_mask_base64"),
                "metrics": data.get("metrics"),
                "scan_type": image_meta.get("scan_type"),
            }
        )

        return {
            "context": context,
            "vision_result": data,
            "progress_event": progress,
        }
    except Exception as exc:
        return {
            "error": f"视觉节点异常：{exc}",
            "progress_event": progress,
        }


async def diagnosis_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context", {})
    image_path = state.get("image_path")
    progress = {"type": "thinking", "msg": "Node 3: 唤醒 Clinical Agent，正在进行病变研判..."}
    emit_stream_event(progress)

    if not image_path:
        return {
            "error": "缺少原始图像，无法执行疾病诊断。",
            "progress_event": progress,
        }

    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    with open(image_path, "rb") as file_obj:
        original_image_base64 = f"data:{mime_type};base64,{base64.b64encode(file_obj.read()).decode('utf-8')}"

    payload = {
        "scan_type": context.get("scan_type"),
        "original_image_base64": original_image_base64,
        "rv_mask_base64": context.get("rv_mask_base64"),
        "faz_mask_base64": context.get("faz_mask_base64"),
        "metrics": context.get("metrics"),
    }

    if not payload["metrics"] or not payload["rv_mask_base64"] or not payload["faz_mask_base64"]:
        return {
            "error": "缺少视觉特征，无法执行疾病诊断。",
            "progress_event": progress,
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(f"{TOOL_BACKEND_URL}/api/v1/agent/classify", json=payload)

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "分类服务调用失败")
            except Exception:
                detail = response.text
            return {
                "error": f"诊断服务返回异常：{detail}",
                "progress_event": progress,
            }

        data = response.json()
        disease_label = data.get("prediction", {}).get("label_cn", "未识别")

        context = dict(context)
        context["diagnosis_result"] = data
        context["last_disease_label"] = disease_label

        return {
            "context": context,
            "diagnosis_result": data,
            "progress_event": progress,
        }
    except Exception as exc:
        return {
            "error": f"诊断节点异常：{exc}",
            "progress_event": progress,
        }


async def rag_node(state: AgentState) -> Dict[str, Any]:
    query = build_rag_query(state)
    progress = {"type": "thinking", "msg": f"Node 4: 唤醒 RAG Agent，正在查阅知识库：{query}"}
    emit_stream_event(progress)

    try:
        docs = await asyncio.to_thread(vector_db.similarity_search, query, 3)
        doc_texts = [doc.page_content for doc in docs]
        rag_result = "\n\n".join([f"【文献参考 {idx + 1}】{text}" for idx, text in enumerate(doc_texts)])
        return {
            "rag_docs": doc_texts,
            "rag_result": rag_result,
            "progress_event": progress,
        }
    except Exception as exc:
        return {
            "error": f"知识检索失败：{exc}",
            "progress_event": progress,
        }


async def final_node(state: AgentState) -> Dict[str, Any]:
    progress = {"type": "thinking", "msg": "Node 5: 正在整合结果并生成最终答复..."}
    emit_stream_event(progress)

    summary_payload = {
        "用户本轮问题": state.get("user_text"),
        "工作流意图": state.get("user_intent"),
        "视觉指标": state.get("context", {}).get("metrics"),
        "诊断结果": state.get("context", {}).get("diagnosis_result"),
        "RAG文献": state.get("rag_result"),
        "blocked": state.get("blocked", False),
        "error": state.get("error"),
    }

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"以下是结构化工作流输出，请基于这些结果回答用户：\n"
        f"{json.dumps(summary_payload, ensure_ascii=False, indent=2)}\n\n"
        "请直接输出最终中文答复，不要输出 JSON。"
    )

    try:
        final_text = await call_glm(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            model=GLM_REPORT_MODEL,
            temperature=0.2,
        )
        if not final_text:
            raise ValueError("最终生成内容为空")
    except Exception:
        final_text = render_non_llm_final(state)

    new_history = list(state.get("history", []))
    new_history.append({"role": "assistant", "content": final_text})
    new_history = keep_recent_history(new_history)

    return {
        "final_text": final_text,
        "history": new_history,
        "progress_event": progress,
    }


# =========================
# LangGraph routes
# =========================
def route_after_intent(state: AgentState) -> str:
    if state.get("error"):
        return "final_node"

    if state.get("needs_vision"):
        return "vision_node"

    if state.get("needs_rag"):
        return "rag_node"

    return "final_node"


def route_after_vision(state: AgentState) -> str:
    if state.get("error") or state.get("blocked"):
        return "final_node"

    if state.get("needs_diagnosis"):
        return "diagnosis_node"

    return "final_node"


def route_after_diagnosis(state: AgentState) -> str:
    if state.get("error"):
        return "final_node"

    if state.get("needs_rag"):
        return "rag_node"

    return "final_node"


# =========================
# Build graph
# =========================
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent_node", intent_node)
    builder.add_node("vision_node", vision_node)
    builder.add_node("diagnosis_node", diagnosis_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("final_node", final_node)

    builder.add_edge(START, "intent_node")
    builder.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "vision_node": "vision_node",
            "rag_node": "rag_node",
            "final_node": "final_node",
        },
    )
    builder.add_conditional_edges(
        "vision_node",
        route_after_vision,
        {
            "diagnosis_node": "diagnosis_node",
            "final_node": "final_node",
        },
    )
    builder.add_conditional_edges(
        "diagnosis_node",
        route_after_diagnosis,
        {
            "rag_node": "rag_node",
            "final_node": "final_node",
        },
    )
    builder.add_edge("rag_node", "final_node")
    builder.add_edge("final_node", END)

    return builder.compile()


graph = build_graph()


# =========================
# FastAPI app
# =========================
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "app": APP_TITLE}


@app.post("/api/chat")
async def chat_with_agent(
    text: str = Form(""),
    file: UploadFile = File(None),
    session_id: str = Form(DEFAULT_SESSION_ID),
):
    user_text = (text or "").strip()
    image_path: Optional[str] = None

    session_state = get_session_state(session_id)
    history: List[Dict[str, str]] = list(session_state.get("history", []))
    context: SessionContext = dict(session_state.get("context", {}))

    if file:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        image_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not user_text:
            user_text = "请帮我分析并诊断这张图。"

    if not user_text and not image_path:
        user_text = "你好"

    history.append({"role": "user", "content": user_text})
    history = keep_recent_history(history)
    save_session_state(session_id, {"history": history, "context": context})

    initial_state: AgentState = {
        "session_id": session_id,
        "user_text": user_text,
        "image_path": image_path,
        "history": history,
        "context": context,
        "blocked": False,
        "error": None,
        "final_text": "",
    }

    async def event_generator():
        yield f"data: {json.dumps({'type': 'thinking', 'msg': 'LangGraph 工作流已启动...' }, ensure_ascii=False)}\n\n"

        final_state: AgentState = dict(initial_state)
        custom_stream_seen = False

        async for chunk in iter_graph_stream(initial_state):
            chunk_type = chunk.get("type")
            chunk_data = chunk.get("data")

            if chunk_type == "custom":
                custom_stream_seen = True
                if isinstance(chunk_data, dict) and chunk_data.get("type"):
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                continue

            if chunk_type != "updates" or not isinstance(chunk_data, dict):
                continue

            for node_name, update in chunk_data.items():
                if not isinstance(update, dict):
                    continue

                final_state.update(update)

                progress_event = update.get("progress_event")
                if progress_event and not custom_stream_seen:
                    yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"

                if node_name == "vision_node" and update.get("vision_result"):
                    yield f"data: {json.dumps({'type': 'vision_data', 'data': update['vision_result']}, ensure_ascii=False)}\n\n"

                if node_name == "diagnosis_node" and update.get("diagnosis_result"):
                    yield f"data: {json.dumps({'type': 'classify_data', 'data': update['diagnosis_result']}, ensure_ascii=False)}\n\n"

                if node_name == "rag_node" and update.get("rag_result"):
                    yield f"data: {json.dumps({'type': 'rag_data', 'text': update['rag_result']}, ensure_ascii=False)}\n\n"

                if node_name == "final_node" and update.get("final_text"):
                    yield f"data: {json.dumps({'type': 'final_text', 'text': update['final_text']}, ensure_ascii=False)}\n\n"

        saved_state = {
            "history": final_state.get("history", history),
            "context": final_state.get("context", context),
        }
        save_session_state(session_id, saved_state)

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("langgraph_octa_app:app", host="0.0.0.0", port=8001, reload=True)
