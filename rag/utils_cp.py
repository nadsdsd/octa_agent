import json
import re
import asyncio
import random
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI
from zhipuai import ZhipuAI

try:
    from langgraph.config import get_stream_writer
except Exception:
    get_stream_writer = None

import config
from api_manager import (
    get_llm_provider_order,
    get_provider_config,
    get_service_candidates,
)
from state import AgentState


@lru_cache(maxsize=1)
def get_redis_client():
    return config.get_redis_client()


@lru_cache(maxsize=1)
def get_zhipu_client() -> ZhipuAI:
    provider_cfg = get_provider_config("ZHIPU")
    api_key = provider_cfg.get("api_key")
    if not api_key:
        raise RuntimeError("未配置 ZHIPUAI_API_KEY")
    return ZhipuAI(api_key=api_key)


@lru_cache(maxsize=8)
def get_openai_compatible_client(provider_name: str) -> AsyncOpenAI:
    normalized = provider_name.upper()
    provider_cfg = get_provider_config(normalized)
    api_key = provider_cfg.get("api_key") or "ollama"
    return AsyncOpenAI(base_url=provider_cfg["base_url"], api_key=api_key)


llm_semaphore = asyncio.Semaphore(1)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# key -> (expire_ts, latency_ms)
_probe_cache: Dict[str, Tuple[float, float]] = {}


def _now_ts() -> float:
    return time.time()


def _preview_obj(obj: Any, max_chars: int | None = None) -> str:
    limit = max_chars or getattr(config, "TRACE_RESPONSE_PREVIEW_CHARS", 400)
    try:
        if isinstance(obj, (dict, list)):
            text = json.dumps(obj, ensure_ascii=False)
        else:
            text = str(obj)
    except Exception:
        text = repr(obj)
    return text[:limit]


def emit_stream_event(event: Optional[Dict[str, Any]]) -> None:
    if not event or get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
        if writer is not None:
            writer(event)
    except Exception:
        pass


def log_backend_event(
    *,
    event: str,
    agent: str,
    target: str,
    detail: Dict[str, Any] | None = None,
    emit_trace: bool = True,
) -> None:
    payload = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "agent": agent,
        "target": target,
        "detail": detail or {},
    }

    if getattr(config, "DEBUG_TRACE_ENABLED", True):
        print("[TRACE] " + json.dumps(payload, ensure_ascii=False))

    if emit_trace:
        emit_stream_event(
            {
                "type": "api_trace",
                "event": event,
                "agent": agent,
                "target": target,
                "detail": detail or {},
            }
        )


async def _probe_url_latency(name: str, url: str, headers: Dict[str, str] | None = None) -> float:
    if not url:
        return float("inf")

    cache_key = f"{name}:{url}"
    cached = _probe_cache.get(cache_key)
    now = _now_ts()
    ttl = getattr(config, "PROBE_CACHE_TTL_SECONDS", 60)

    if cached and cached[0] > now:
        return cached[1]

    start = time.perf_counter()
    timeout = getattr(config, "PROBE_TIMEOUT_SECONDS", 2.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers or {})
            if resp.status_code >= 500:
                raise RuntimeError(f"probe failed: {resp.status_code}")
        latency_ms = (time.perf_counter() - start) * 1000.0
    except Exception:
        latency_ms = float("inf")

    _probe_cache[cache_key] = (now + ttl, latency_ms)
    return latency_ms


async def rank_llm_providers(preferred: str | None = None, agent_name: str = "LLM Agent") -> List[str]:
    base_order = get_llm_provider_order(preferred)
    scored: List[Tuple[str, float]] = []

    for provider_name in base_order:
        cfg = get_provider_config(provider_name)
        probe_url = cfg.get("probe_url", "")
        headers = {}
        if cfg.get("api_key") and cfg.get("api_key") != "ollama":
            headers["Authorization"] = f"Bearer {cfg['api_key']}"

        latency = await _probe_url_latency(
            name=f"provider:{provider_name}",
            url=probe_url,
            headers=headers,
        )
        scored.append((provider_name, latency))

    ranked = [name for name, _ in sorted(scored, key=lambda x: x[1])]
    log_backend_event(
        event="provider_ranked",
        agent=agent_name,
        target="llm_router",
        detail={
            "preferred": preferred,
            "ranked": [{"provider": name, "latency_ms": None if latency == float('inf') else round(latency, 2)} for name, latency in sorted(scored, key=lambda x: x[1])],
        },
    )
    return ranked


async def rank_service_candidates(service_name: str, agent_name: str) -> List[Dict[str, Any]]:
    candidates = get_service_candidates(service_name)
    scored: List[Tuple[Dict[str, Any], float]] = []

    for item in candidates:
        latency = await _probe_url_latency(
            name=f"service:{service_name}:{item['name']}",
            url=item.get("probe_url", ""),
        )
        scored.append((item, latency))

    ranked = [item for item, _ in sorted(scored, key=lambda x: x[1])]
    log_backend_event(
        event="service_ranked",
        agent=agent_name,
        target=service_name,
        detail={
            "ranked": [
                {
                    "name": item["name"],
                    "base_url": item["base_url"],
                    "latency_ms": None if latency == float("inf") else round(latency, 2),
                }
                for item, latency in sorted(scored, key=lambda x: x[1])
            ]
        },
    )
    return ranked


async def _call_provider(provider_name: str, messages: List[Dict[str, str]], temperature: float) -> str:
    normalized = provider_name.upper()
    provider_cfg = get_provider_config(normalized)

    if normalized == "ZHIPU":
        def _zhipu() -> str:
            resp = get_zhipu_client().chat.completions.create(
                model=provider_cfg["model"],
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_zhipu)

    client = get_openai_compatible_client(normalized)
    resp = await client.chat.completions.create(
        model=provider_cfg["model"],
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


async def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_retries: int = 3,
    mode: Optional[str] = None,
    agent_name: str = "LLM Agent",
) -> str:
    provider_order = await rank_llm_providers(mode, agent_name=agent_name)
    last_error: Optional[Exception] = None

    log_backend_event(
        event="llm_call_start",
        agent=agent_name,
        target="llm_router",
        detail={
            "requested_mode": mode,
            "provider_order": provider_order,
            "temperature": temperature,
            "message_preview": _preview_obj(messages),
        },
    )

    for provider_name in provider_order:
        for attempt in range(max_retries):
            start = time.perf_counter()
            try:
                log_backend_event(
                    event="llm_provider_try",
                    agent=agent_name,
                    target=provider_name,
                    detail={"attempt": attempt + 1, "max_retries": max_retries},
                )

                async with llm_semaphore:
                    await asyncio.sleep(0.05)
                    result = await _call_provider(provider_name, messages, temperature)

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                log_backend_event(
                    event="llm_provider_success",
                    agent=agent_name,
                    target=provider_name,
                    detail={
                        "attempt": attempt + 1,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "response_preview": _preview_obj(result),
                    },
                )
                return result

            except Exception as exc:
                last_error = exc
                error_str = str(exc)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                log_backend_event(
                    event="llm_provider_error",
                    agent=agent_name,
                    target=provider_name,
                    detail={
                        "attempt": attempt + 1,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "error": error_str,
                    },
                )

                should_retry = provider_name.upper() == "ZHIPU" and (
                    "429" in error_str or "1302" in error_str or "速率限制" in error_str
                )
                if should_retry and attempt < max_retries - 1:
                    wait_time = (2 ** (attempt + 1)) + random.uniform(0.1, 1.0)
                    log_backend_event(
                        event="llm_provider_retry",
                        agent=agent_name,
                        target=provider_name,
                        detail={"wait_time_s": round(wait_time, 2)},
                    )
                    await asyncio.sleep(wait_time)
                    continue
                break

    if last_error:
        raise last_error
    raise RuntimeError("LLM 调用失败，且没有返回可追踪异常。")


async def call_json_api(
    service_name: str,
    *,
    agent_name: str,
    method: str = "POST",
    json_body: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    ranked = await rank_service_candidates(service_name, agent_name=agent_name)
    last_error: Optional[Exception] = None

    for candidate in ranked:
        url = f"{candidate['base_url']}{candidate['path']}"
        start = time.perf_counter()

        try:
            log_backend_event(
                event="service_call_start",
                agent=agent_name,
                target=url,
                detail={
                    "service_name": service_name,
                    "method": method.upper(),
                    "json_preview": _preview_obj(json_body) if json_body else None,
                    "has_files": bool(files),
                },
            )

            async with httpx.AsyncClient(timeout=timeout or config.HTTP_TIMEOUT_SECONDS) as client:
                if method.upper() == "POST":
                    resp = await client.post(url, json=json_body, files=files)
                elif method.upper() == "GET":
                    resp = await client.get(url, params=json_body)
                else:
                    raise ValueError(f"Unsupported method: {method}")

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            preview = None
            data = None
            try:
                data = resp.json()
                preview = _preview_obj(data)
            except Exception:
                preview = _preview_obj(resp.text)

            log_backend_event(
                event="service_call_response",
                agent=agent_name,
                target=url,
                detail={
                    "service_name": service_name,
                    "status_code": resp.status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "response_preview": preview,
                },
            )

            resp.raise_for_status()

            if data is None:
                data = resp.json()
            return data

        except Exception as exc:
            last_error = exc
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            log_backend_event(
                event="service_call_error",
                agent=agent_name,
                target=url,
                detail={
                    "service_name": service_name,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": str(exc),
                },
            )
            continue

    if last_error:
        raise last_error
    raise RuntimeError(f"{service_name} 调用失败，且没有可追踪异常。")


def get_session_state(session_id: str) -> Dict[str, Any]:
    raw = get_redis_client().get(f"agent_session:{session_id}")
    if raw:
        return json.loads(raw)
    return {"history": [{"role": "system", "content": config.SYSTEM_PROMPT}], "context": {}}


def save_session_state(session_id: str, state: Dict[str, Any]) -> None:
    get_redis_client().setex(
        f"agent_session:{session_id}",
        config.SESSION_TTL_SECONDS,
        json.dumps(state, ensure_ascii=False),
    )


def keep_recent_history(history: List[Dict[str, str]], max_turns: int = 15) -> List[Dict[str, str]]:
    if len(history) <= max_turns:
        return history
    return [history[0]] + history[-(max_turns - 1):]


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
        json_str = match.group(0)
        json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
        try:
            return json.loads(json_str)
        except Exception as exc:
            raise ValueError(f"JSON 清洗后仍无法解析: {exc}") from exc


def normalize_stream_chunk(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict) and "type" in chunk and "data" in chunk:
        return chunk
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return {"type": chunk[0], "data": chunk[1]}
    if isinstance(chunk, dict):
        return {"type": "updates", "data": chunk}
    return {"type": "unknown", "data": chunk}


async def iter_graph_stream(graph_instance, initial_state: AgentState):
    stream_attempts = [
        {"stream_mode": ["custom", "updates"], "version": "v2"},
        {"stream_mode": "updates", "version": "v2"},
        {"stream_mode": "updates"},
    ]
    last_type_error: Optional[Exception] = None
    for kwargs in stream_attempts:
        try:
            async for raw_chunk in graph_instance.astream(initial_state, **kwargs):
                yield normalize_stream_chunk(raw_chunk)
            return
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error:
        raise last_type_error


def convert_metrics_to_physical(metrics: Dict[str, Any], scan_type: str = "") -> Dict[str, Any]:
    if not metrics:
        return {}
    fov_mm = 3.0 if "3x3" in str(scan_type).lower() else 6.0
    img_px = 224.0
    mm_per_px = fov_mm / img_px
    mm2_per_px2 = mm_per_px ** 2
    inv_mm_per_inv_px = img_px / fov_mm

    return {
        "faz_area_mm2": round(metrics.get("faz_area_px", 0) * mm2_per_px2, 4),
        "faz_perim_mm": round(metrics.get("faz_perim_px", 0) * mm_per_px, 4),
        "faz_circularity": round(metrics.get("faz_circularity", 0), 4),
        "rv_density": round(metrics.get("rv_density", 0), 4),
        "rv_flow_area_mm2": round(metrics.get("rv_flow_area_px", 0) * mm2_per_px2, 4),
        "rv_line_density_mm-1": round(metrics.get("rv_line_density_px-1", 0) * inv_mm_per_inv_px, 4),
        "rv_branch_points": round(metrics.get("rv_branch_points", 0), 4),
        "faz300_sim_density": round(metrics.get("faz300_sim_density", 0), 4),
    }


def append_warning(state: Dict[str, Any], message: str) -> List[str]:
    warnings = list(state.get("warnings", []))
    warnings.append(message)
    return warnings