import base64
import mimetypes

import httpx
from typing import Dict, Any

from state import AgentState
from config import HTTP_TIMEOUT_SECONDS
from api_manager import get_service_url
from utils import emit_stream_event


LABEL_DISPLAY = {
    "NORMAL": "正常 (NORMAL)",
    "AMD": "老年性黄斑变性 (AMD)",
    "DR": "糖尿病视网膜病变 (DR)",
    "CNV": "脉络膜新生血管 (CNV)",
    "CSC": "中央浆液性脉络膜视网膜病变 (CSC)",
    "RVO": "视网膜静脉阻塞 (RVO)",
    "OTHERS": "其他 (OTHERS)",
}


def encode_file_to_data_url(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    with open(file_path, "rb") as file_obj:
        encoded = base64.b64encode(file_obj.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


async def diagnosis_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context", {})
    image_path = state.get("image_path")
    progress = {"type": "thinking", "msg": "Node 3: 唤醒 Clinical Agent，正在进行病变研判..."}
    emit_stream_event(progress)

    if not image_path:
        return {"error": "缺少原始图像，无法执行疾病诊断。", "progress_event": progress}

    payload = {
        "scan_type": context.get("scan_type"),
        "original_image_base64": encode_file_to_data_url(image_path),
        "rv_mask_base64": context.get("rv_mask_base64"),
        "faz_mask_base64": context.get("faz_mask_base64"),
        "metrics": context.get("metrics"),
    }

    if not payload["metrics"] or not payload["rv_mask_base64"] or not payload["faz_mask_base64"]:
        return {"error": "缺少视觉特征，无法执行疾病诊断。", "progress_event": progress}

    details = [
        {
            "type": "thinking_detail",
            "agent": "Clinical Agent",
            "title": "深度特征映射",
            "content": "已接收 Vision Agent 传递的高维视觉特征与掩码矩阵。正在启动图像多模态分类网络进行前向传播与病理特征空间映射...",
        }
    ]

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as http_client:
            response = await http_client.post(get_service_url("diagnosis_classify"), json=payload)

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "分类服务调用失败")
            except Exception:
                detail = response.text
            return {"error": f"诊断服务返回异常：{detail}", "progress_event": progress}

        data = response.json()
        prediction = data.get("prediction", {}) or {}
        disease_label = str(prediction.get("label_en") or prediction.get("label_cn") or "UNKNOWN").upper().strip()
        disease_display = LABEL_DISPLAY.get(disease_label, disease_label)
        image_confidence = prediction.get("confidence")

        details.append(
            {
                "type": "thinking_detail",
                "agent": "Clinical Agent",
                "title": "分类网络推理完成",
                "content": (
                    f"模型全连接层输出分类概率分布。当前置信度最高（Top-1）的判定类别为：{disease_display}。"
                    f"{f' 分类置信度约为 {float(image_confidence):.3f}。' if image_confidence is not None else ''}"
                ),
            }
        )

        new_context = dict(context)
        new_context["diagnosis_result"] = data
        new_context["image_diagnosis_label"] = disease_label
        if image_confidence is not None:
            try:
                new_context["image_confidence"] = float(image_confidence)
            except Exception:
                pass
        new_context.setdefault("last_disease_label", disease_label)

        return {
            "context": new_context,
            "diagnosis_result": data,
            "progress_event": {"type": "thinking", "msg": f"Node 3: 病变研判完成，结果为 {disease_display}"},
            "thinking_details": details,
        }
    except Exception as exc:
        return {"error": f"诊断节点异常：{exc}", "progress_event": progress}
