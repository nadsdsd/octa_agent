import os
import json
import shutil
import uuid
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import APP_TITLE, UPLOAD_DIR, DEFAULT_SESSION_ID, build_enabled_agents
from state import AgentState, SessionContext
from utils import (
    get_session_state,
    save_session_state,
    keep_recent_history,
    iter_graph_stream,
    get_redis_client,
)
from graph import agent_graph

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_LOCK_TTL_SECONDS = int(os.getenv("CHAT_LOCK_TTL_SECONDS", "900"))

# 新图上传时，需要清掉上一轮图像分析链路留下的上下文
ANALYSIS_CONTEXT_KEYS_TO_CLEAR = {
    "rv_mask_base64",
    "faz_mask_base64",
    "metrics",
    "scan_type",
    "diagnosis_result",
    "last_disease_label",
    "physical_metrics",
    "image_diagnosis_label",
    "physical_diagnosis_label",
    "pathology_diagnosis_result",
}


def clear_analysis_context(context: SessionContext) -> SessionContext:
    new_context: SessionContext = dict(context)
    for key in ANALYSIS_CONTEXT_KEYS_TO_CLEAR:
        new_context.pop(key, None)
    new_context["symptoms_requested"] = False
    return new_context


def acquire_session_lock(session_id: str) -> Optional[str]:
    token = uuid.uuid4().hex
    lock_key = f"agent_chat_lock:{session_id}"
    ok = get_redis_client().set(lock_key, token, nx=True, ex=CHAT_LOCK_TTL_SECONDS)
    return token if ok else None


def release_session_lock(session_id: str, token: Optional[str]) -> None:
    if not token:
        return
    lock_key = f"agent_chat_lock:{session_id}"
    lua = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('DEL', KEYS[1])
    else
        return 0
    end
    """
    try:
        get_redis_client().eval(lua, 1, lock_key, token)
    except Exception:
        pass


def refresh_session_lock(session_id: str, token: Optional[str]) -> None:
    if not token:
        return
    lock_key = f"agent_chat_lock:{session_id}"
    lua = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('EXPIRE', KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    try:
        get_redis_client().eval(lua, 1, lock_key, token, CHAT_LOCK_TTL_SECONDS)
    except Exception:
        pass


def make_busy_stream(message: str) -> StreamingResponse:
    async def _busy_generator():
        yield f"data: {json.dumps({'type': 'error', 'msg': message}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_busy_generator(), media_type="text/event-stream")


def merge_state(final_state: AgentState, update: Dict[str, Any]) -> AgentState:
    merged = dict(final_state)

    # context 深合并
    if isinstance(update.get("context"), dict):
        ctx = dict(merged.get("context", {}))
        ctx.update(update["context"])
        merged["context"] = ctx

    # agent_status 深合并
    if isinstance(update.get("agent_status"), dict):
        status_map = dict(merged.get("agent_status", {}))
        status_map.update(update["agent_status"])
        merged["agent_status"] = status_map

    # 其他字段直接覆盖
    for key, value in update.items():
        if key in {"context", "agent_status"}:
            continue
        merged[key] = value

    return merged


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "app": APP_TITLE}


@app.post("/api/chat")
async def chat_with_agent(
    text: str = Form(""),
    file: UploadFile = File(None),
    session_id: str = Form(DEFAULT_SESSION_ID),
):
    # 会话级并发锁：同一 session_id 同时只允许一个请求
    lock_token = acquire_session_lock(session_id)
    if not lock_token:
        return make_busy_stream("当前会话已有一条消息正在处理中，请等待本轮分析完成后再发送下一条。")

    user_text = (text or "").strip()
    image_path: Optional[str] = None

    session_state = get_session_state(session_id)
    history: List[Dict[str, str]] = list(session_state.get("history", []))
    context: SessionContext = dict(session_state.get("context", {}))

    try:
        if file:
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            safe_name = os.path.basename(file.filename or "upload_image")
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            image_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_name))

            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 新图 = 新一轮图像分析，清理旧结果，避免复用上一轮诊断/指标
            context = clear_analysis_context(context)

            if not user_text:
                user_text = "请帮我分析并诊断这张图。"

        if not user_text and not image_path:
            user_text = "你好"

        history.append({"role": "user", "content": user_text})
        history = keep_recent_history(history)

        # 先保存用户输入后的最新会话状态
        save_session_state(session_id, {"history": history, "context": context})

        initial_state: AgentState = {
            "session_id": session_id,
            "user_text": user_text,
            "image_path": image_path,
            "history": history,
            "context": context,
            "enabled_agents": build_enabled_agents(),
            "agent_status": {},
            "warnings": [],
            "blocked": False,
            "error": None,
            "final_text": "",
        }

        async def event_generator():
            final_state: AgentState = dict(initial_state)
            custom_stream_seen = False

            try:
                yield f"data: {json.dumps({'type': 'thinking', 'msg': 'LangGraph 工作流已启动...'}, ensure_ascii=False)}\n\n"

                async for chunk in iter_graph_stream(agent_graph, initial_state):
                    refresh_session_lock(session_id, lock_token)

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

                        final_state = merge_state(final_state, update)

                        progress_event = update.get("progress_event")
                        if progress_event and not custom_stream_seen:
                            yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"

                        for detail in update.get("thinking_details", []) or []:
                            yield f"data: {json.dumps(detail, ensure_ascii=False)}\n\n"

                        if update.get("warnings"):
                            warning = update["warnings"][-1]
                            yield f"data: {json.dumps({'type': 'warning', 'msg': warning}, ensure_ascii=False)}\n\n"

                        if node_name == "vision_node" and update.get("vision_result"):
                            yield f"data: {json.dumps({'type': 'vision_data', 'data': update['vision_result']}, ensure_ascii=False)}\n\n"

                        if node_name == "diagnosis_node" and update.get("diagnosis_result"):
                            yield f"data: {json.dumps({'type': 'classify_data', 'data': update['diagnosis_result']}, ensure_ascii=False)}\n\n"

                        if node_name == "physical_node" and update.get("physical_result"):
                            yield f"data: {json.dumps({'type': 'physical_data', 'data': update['physical_result']}, ensure_ascii=False)}\n\n"

                        if node_name == "pathology_node" and update.get("pathology_result"):
                            yield f"data: {json.dumps({'type': 'pathology_data', 'data': update['pathology_result']}, ensure_ascii=False)}\n\n"

                        if node_name == "rag_node" and update.get("rag_result") is not None:
                            yield f"data: {json.dumps({'type': 'rag_data', 'text': update.get('rag_result', '')}, ensure_ascii=False)}\n\n"

                        if node_name == "final_node" and update.get("final_text"):
                            yield f"data: {json.dumps({'type': 'final_text', 'text': update['final_text']}, ensure_ascii=False)}\n\n"

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                final_state["error"] = f"主流程异常：{exc}"
                yield f"data: {json.dumps({'type': 'error', 'msg': str(exc)}, ensure_ascii=False)}\n\n"
            finally:
                saved_state = {
                    "history": final_state.get("history", history),
                    "context": final_state.get("context", context),
                }
                try:
                    save_session_state(session_id, saved_state)
                finally:
                    release_session_lock(session_id, lock_token)

                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception:
        release_session_lock(session_id, lock_token)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)