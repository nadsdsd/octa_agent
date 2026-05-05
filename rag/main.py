import os
import json
import shutil
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import Depends, FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from auth_bridge import get_current_user, require_chat_user
from config import APP_TITLE, UPLOAD_DIR, build_enabled_agents
from database import Base, engine
from session_memory import session_memory_store
from state import AgentState, SessionContext
from user_models import User, UserSessionMemory  # noqa: F401
from utils import (
    keep_recent_history,
    iter_graph_stream,
    get_redis_client,
    set_stream_event_queue,
    reset_stream_event_queue,
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
    "pending_report_after_pathology",
}


@app.on_event("startup")
def startup_init_db() -> None:
    Base.metadata.create_all(bind=engine)


class CurrentUserResponse(dict):
    pass


def clear_analysis_context(context: SessionContext) -> SessionContext:
    new_context: SessionContext = dict(context)
    for key in ANALYSIS_CONTEXT_KEYS_TO_CLEAR:
        new_context.pop(key, None)
    new_context["symptoms_requested"] = False
    return new_context


def acquire_session_lock(session_id: str) -> Optional[str]:
    import uuid
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

    if isinstance(update.get("context"), dict):
        ctx = dict(merged.get("context", {}))
        ctx.update(update["context"])
        merged["context"] = ctx

    if isinstance(update.get("agent_status"), dict):
        status_map = dict(merged.get("agent_status", {}))
        status_map.update(update["agent_status"])
        merged["agent_status"] = status_map

    for key, value in update.items():
        if key in {"context", "agent_status"}:
            continue
        merged[key] = value

    return merged


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "app": APP_TITLE}


@app.get("/auth/me")
async def auth_me(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role,
        "session_id": session_memory_store.build_session_id(current_user.id),
    }


@app.post("/api/chat")
async def chat_with_agent(
    text: str = Form(""),
    file: UploadFile = File(None),
    # 为了兼容旧前端，这个字段保留，但后端不再信任它作为会话归属
    session_id: Optional[str] = Form(None),
    current_user: User = Depends(require_chat_user),
):
    effective_session_id = session_memory_store.build_session_id(current_user.id)

    lock_token = acquire_session_lock(effective_session_id)
    if not lock_token:
        return make_busy_stream("当前账号已有一条消息正在处理中，请等待本轮分析完成后再发送下一条。")

    user_text = (text or "").strip()
    image_path: Optional[str] = None

    session_state = session_memory_store.load_user_state(current_user.id)
    history: List[Dict[str, str]] = list(session_state.get("history", []))
    context: SessionContext = dict(session_state.get("context", {}))

    try:
        if file:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            safe_name = os.path.basename(file.filename or "upload_image")
            unique_name = f"{effective_session_id}_{safe_name}"
            image_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_name))

            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            context = clear_analysis_context(context)
            if not user_text:
                user_text = "请帮我分析并诊断这张图。"

        if not user_text and not image_path:
            user_text = "你好"

        history.append({"role": "user", "content": user_text})
        history = keep_recent_history(history)

        session_memory_store.save_user_state(current_user.id, {"history": history, "context": context})

        initial_state: AgentState = {
            "session_id": effective_session_id,
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
            pending_event_queue: asyncio.Queue = asyncio.Queue()
            stream_queue_token = set_stream_event_queue(pending_event_queue)
            seen_event_keys: set[str] = set()

            def make_event_key(payload: Dict[str, Any]) -> str:
                try:
                    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
                except Exception:
                    return repr(payload)

            async def emit_pending_runtime_events():
                while True:
                    try:
                        runtime_event = pending_event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not isinstance(runtime_event, dict) or not runtime_event.get("type"):
                        continue
                    event_key = make_event_key(runtime_event)
                    if event_key in seen_event_keys:
                        continue
                    seen_event_keys.add(event_key)
                    yield f"data: {json.dumps(runtime_event, ensure_ascii=False)}\n\n"

            async def graph_chunk_producer(output_queue: asyncio.Queue):
                try:
                    async for chunk in iter_graph_stream(agent_graph, initial_state):
                        await output_queue.put(("chunk", chunk))
                except Exception as exc:
                    await output_queue.put(("error", exc))
                finally:
                    await output_queue.put(("done", None))

            try:
                yield f"data: {json.dumps({'type': 'thinking', 'msg': 'LangGraph 工作流已启动...'}, ensure_ascii=False)}\n\n"
                chunk_queue: asyncio.Queue = asyncio.Queue()
                producer_task = asyncio.create_task(graph_chunk_producer(chunk_queue))
                stream_done = False

                while not stream_done:
                    refresh_session_lock(effective_session_id, lock_token)
                    pending_get = asyncio.create_task(pending_event_queue.get())
                    chunk_get = asyncio.create_task(chunk_queue.get())
                    done, pending = await asyncio.wait(
                        {pending_get, chunk_get},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for pending_task in pending:
                        pending_task.cancel()

                    completed_task = next(iter(done))
                    source, payload = (None, None)
                    if completed_task is pending_get:
                        runtime_event = completed_task.result()
                        if isinstance(runtime_event, dict) and runtime_event.get("type"):
                            event_key = make_event_key(runtime_event)
                            if event_key not in seen_event_keys:
                                seen_event_keys.add(event_key)
                                yield f"data: {json.dumps(runtime_event, ensure_ascii=False)}\n\n"
                        async for buffered_event in emit_pending_runtime_events():
                            yield buffered_event
                        continue
                    else:
                        source, payload = completed_task.result()
                        async for buffered_event in emit_pending_runtime_events():
                            yield buffered_event

                    if source == "done":
                        stream_done = True
                        continue

                    if source == "error":
                        raise payload

                    chunk = payload

                    chunk_type = chunk.get("type")
                    chunk_data = chunk.get("data")

                    if chunk_type == "custom":
                        custom_stream_seen = True
                        if isinstance(chunk_data, dict) and chunk_data.get("type"):
                            event_key = make_event_key(chunk_data)
                            if event_key not in seen_event_keys:
                                seen_event_keys.add(event_key)
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
                await producer_task

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                final_state["error"] = f"主流程异常：{exc}"
                yield f"data: {json.dumps({'type': 'error', 'msg': str(exc)}, ensure_ascii=False)}\n\n"
            finally:
                reset_stream_event_queue(stream_queue_token)
                saved_state = {
                    "history": final_state.get("history", history),
                    "context": final_state.get("context", context),
                }
                try:
                    session_memory_store.save_user_state(current_user.id, saved_state)
                finally:
                    release_session_lock(effective_session_id, lock_token)

                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception:
        release_session_lock(effective_session_id, lock_token)
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
