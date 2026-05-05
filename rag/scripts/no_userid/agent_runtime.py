from __future__ import annotations

from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, Tuple

import config
from state import AgentState


NodeCallable = Callable[[AgentState], Awaitable[Dict[str, Any]]]

REQUIRED_OUTPUTS = {
    "intent_node": [("user_intent",)],
    "vision_node": [("context", "metrics")],
    "diagnosis_node": [("context", "image_diagnosis_label")],
    "physical_node": [("context", "physical_metrics")],
    "pathology_node": [("context", "pathology_diagnosis_result")],
    "rag_node": [("rag_docs",)],
    "final_node": [("final_text",)],
}


def _merge_state_preview(state: AgentState, update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state)
    merged.update(update)
    merged_context = dict(state.get("context", {}))
    merged_context.update(update.get("context", {}))
    merged["context"] = merged_context
    return merged


def _get_nested(data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _build_status_update(state: AgentState, node_name: str, status: str, **extra: Any) -> Dict[str, Any]:
    agent_status = deepcopy(state.get("agent_status", {}))
    current = dict(agent_status.get(node_name, {}))
    current.setdefault("attempts", 0)
    current["attempts"] += 1
    current["status"] = status
    current.update(extra)
    agent_status[node_name] = current
    return agent_status


def _append_warning(state: AgentState, message: str) -> list[str]:
    warnings = list(state.get("warnings", []))
    warnings.append(message)
    return warnings


def validate_node_output(node_name: str, state: AgentState, update: Dict[str, Any]) -> tuple[bool, str | None]:
    if update.get("blocked"):
        return True, None

    merged = _merge_state_preview(state, update)
    required_paths = REQUIRED_OUTPUTS.get(node_name, [])
    missing = []
    for path in required_paths:
        value = _get_nested(merged, path)
        if value is None or value == "":
            missing.append(".".join(path))

    if not missing:
        return True, None
    return False, f"{node_name} 缺少关键输出: {', '.join(missing)}"


def guarded_node(node_name: str, node_callable: NodeCallable) -> NodeCallable:
    async def _runner(state: AgentState) -> Dict[str, Any]:
        enabled_agents = dict(state.get("enabled_agents") or config.build_enabled_agents())
        if not enabled_agents.get(node_name, True):
            return {
                "enabled_agents": enabled_agents,
                "agent_status": _build_status_update(
                    state,
                    node_name,
                    "skipped",
                    output_valid=False,
                    reason="agent ablation disabled",
                    skipped_by_ablation=True,
                ),
                "warnings": _append_warning(state, f"{node_name} 已按消融配置跳过。"),
                "last_node": node_name,
            }

        try:
            update = await node_callable(state)
            update = update or {}

            if update.get("error") and node_name != "final_node":
                warning = f"{node_name} 失败，已自动降级: {update['error']}"
                update.pop("error", None)
                update["warnings"] = _append_warning(state, warning)
                update["agent_status"] = _build_status_update(
                    state,
                    node_name,
                    "failed",
                    output_valid=False,
                    last_error=warning,
                    reason="runtime error downgraded to warning",
                )
                update["enabled_agents"] = enabled_agents
                update["last_node"] = node_name
                return update

            output_valid, reason = validate_node_output(node_name, state, update)
            update["agent_status"] = _build_status_update(
                state,
                node_name,
                "success" if output_valid else "degraded",
                output_valid=output_valid,
                reason=reason or "ok",
            )
            if reason and node_name != "final_node":
                update["warnings"] = _append_warning(state, reason)
            update["enabled_agents"] = enabled_agents
            update["last_node"] = node_name
            return update
        except Exception as exc:
            warning = f"{node_name} 异常，已进入兜底路由: {exc}"
            return {
                "enabled_agents": enabled_agents,
                "agent_status": _build_status_update(
                    state,
                    node_name,
                    "failed",
                    output_valid=False,
                    last_error=str(exc),
                    reason="exception caught by guarded_node",
                ),
                "warnings": _append_warning(state, warning),
                "last_node": node_name,
            }

    return _runner


def route_next_step(state: AgentState) -> str:
    context = state.get("context", {})
    enabled_agents = dict(state.get("enabled_agents") or config.build_enabled_agents())
    agent_status = state.get("agent_status", {})
    has_new_image = bool(state.get("has_new_image"))

    def failed(node_name: str) -> bool:
        return agent_status.get(node_name, {}).get("status") in {"failed", "skipped"}

    if state.get("blocked"):
        return "final_node"

    if state.get("needs_pathology") and enabled_agents.get("pathology_node", True) and not context.get("pathology_diagnosis_result") and not failed("pathology_node"):
        return "pathology_node"

    has_metrics = bool(context.get("metrics"))
    has_masks = bool(context.get("rv_mask_base64") and context.get("faz_mask_base64"))

    if state.get("needs_vision") and enabled_agents.get("vision_node", True) and state.get("image_path") and not failed("vision_node"):
        if has_new_image or not has_metrics:
            return "vision_node"

    if state.get("needs_diagnosis") and enabled_agents.get("diagnosis_node", True) and not failed("diagnosis_node"):
        if has_metrics and has_masks and (has_new_image or not context.get("image_diagnosis_label")):
            return "diagnosis_node"

    if state.get("needs_physical") and enabled_agents.get("physical_node", True) and not failed("physical_node"):
        if has_metrics and (has_new_image or not context.get("physical_metrics") or not context.get("physical_diagnosis_label")):
            return "physical_node"

    img_label = context.get("image_diagnosis_label")
    phys_label = context.get("physical_diagnosis_label")
    if img_label and phys_label and img_label != phys_label and not context.get("symptoms_requested") and not context.get("pathology_diagnosis_result"):
        return "final_node"

    if state.get("needs_rag") and enabled_agents.get("rag_node", True) and not failed("rag_node"):
        if not state.get("rag_docs"):
            return "rag_node"

    return "final_node"
