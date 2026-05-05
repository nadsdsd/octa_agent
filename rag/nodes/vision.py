import httpx
from typing import Dict, Any

from state import AgentState
from config import HTTP_TIMEOUT_SECONDS
from api_manager import get_service_url
from utils import emit_stream_event


async def vision_node(state: AgentState) -> Dict[str, Any]:
    image_path = state.get("image_path")
    if not image_path:
        return {"error": "vision 节点被触发，但没有 image_path。"}

    progress = {"type": "thinking", "msg": "Node 2: 唤醒 Vision Agent，正在提取 OCTA 分割结果与血流指标..."}
    emit_stream_event(progress)

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as http_client:
            with open(image_path, "rb") as file_obj:
                response = await http_client.post(
                    get_service_url("vision_analyze"),
                    files={"file": file_obj},
                )

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "视觉服务调用失败")
            except Exception:
                detail = response.text
            return {"error": f"视觉服务返回异常：{detail}", "progress_event": progress}

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
            "progress_event": {"type": "thinking", "msg": "Node 2: 图像分割与指标提取完成。"},
        }
    except Exception as exc:
        return {"error": f"视觉节点异常：{exc}", "progress_event": progress}
