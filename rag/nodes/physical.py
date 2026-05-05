import json
import re
from typing import Dict, Any

from state import AgentState
from config import DISEASE_PHYSICAL_STATS, PHYSICAL_LLM_PROVIDER
from utils import call_llm, emit_stream_event, convert_metrics_to_physical, extract_json_object


VALID_PHYSICAL_LABELS = {"AMD", "CNV", "CSC", "DR", "NORMAL", "RVO", "OTHERS"}


def _allowed_labels_for_scan(scan_type: str) -> set[str]:
    scan_type_upper = str(scan_type or "").upper()
    if scan_type_upper.startswith("3"):
        return {"NORMAL", "AMD", "DR", "CNV"}
    return {"NORMAL", "AMD", "DR", "CNV", "CSC", "RVO", "OTHERS"}


def _parse_physical_response(raw_text: str, allowed_labels: set[str]) -> Dict[str, str]:
    try:
        parsed = extract_json_object(raw_text)
        label = str(parsed.get("label", "")).upper().strip()
        reasoning = str(parsed.get("thought_process", "")).strip()
        if label in VALID_PHYSICAL_LABELS and label in allowed_labels:
            return {
                "label": label,
                "thought_process": reasoning or "模型已返回标签，但未提供详细推理。",
            }
    except Exception:
        pass

    text = (raw_text or "").replace("\r", " ").strip()
    label_match = re.search(r"\b(AMD|CNV|CSC|DR|NORMAL|RVO|OTHERS)\b", text.upper())
    if not label_match:
        raise ValueError("未能从数值分析输出中提取有效疾病标签")

    label = label_match.group(1)
    if label not in allowed_labels:
        raise ValueError(f"数值分析输出标签 {label} 与当前 scan_type 不匹配")
    reasoning = text
    reasoning = re.sub(r"(?i)\b(label|标签)\b\s*[:：]?\s*" + re.escape(label), "", reasoning).strip(" \n:：,-")
    if not reasoning:
        reasoning = f"模型输出中识别到最终标签为 {label}，但未提供稳定的详细推理文本。"

    return {"label": label, "thought_process": reasoning}


async def physical_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context", {})
    progress = {"type": "thinking", "msg": "Node 3B: 唤醒 Physical Agent，正在进行体检数值统计学比对..."}
    emit_stream_event(progress)

    metrics = context.get("metrics", {})
    scan_type = context.get("scan_type", "macula_6x6")

    if not metrics:
        return {"error": "缺少 metrics，无法执行数值分析。", "progress_event": progress}

    details = []
    physical_metrics = convert_metrics_to_physical(metrics, scan_type)
    allowed_labels = _allowed_labels_for_scan(scan_type)
    allowed_labels_text = ", ".join(sorted(allowed_labels))

    details.append(
        {
            "type": "thinking_detail",
            "agent": "Physical Agent",
            "title": "1. 提取并转换物理尺寸参数",
            "content": f"FOV设定: {scan_type}\n转换后物理指标:\n{json.dumps(physical_metrics, indent=2, ensure_ascii=False)}",
        }
    )

    prompt = (
        f"你是眼底病变体检数值分析师。目前处理的FOV尺寸类型为: {scan_type}。\n"
        f"这是患者当前转换后的 OCTA 物理指标：\n{json.dumps(physical_metrics, indent=2, ensure_ascii=False)}\n\n"
        f"请参考以下各疾病的统计学分布 (Mean ± Std)：\n{DISEASE_PHYSICAL_STATS}\n\n"
        f"任务：请基于当前患者的各项数值偏离度，选出最符合的一个疾病标签。当前 scan_type 允许的标签只有：{allowed_labels_text}。\n"
        "如果当前是 6M，且这些常见病种都不匹配，可以输出 OTHERS；如果当前是 3M，绝对不能输出 CSC、RVO、OTHERS。\n"
        "请优先按以下 JSON 输出：\n"
        '{"thought_process":"详细写出你的比对过程，指出哪些核心指标符合哪种疾病的均值范围，排除了哪些疾病","label":"具体的疾病缩写标签"}\n'
        f"如果你无法稳定输出 JSON，也必须保证答案中明确出现且只选择一个最终标签：{allowed_labels_text}。"
    )

    try:
        response = await call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            mode=PHYSICAL_LLM_PROVIDER,
            agent_name="physical_agent"
        )
        parsed = _parse_physical_response(response, allowed_labels)

        phys_label = parsed.get("label", "UNKNOWN").upper()
        reasoning = parsed.get("thought_process", "无详细推理过程。")

        details.append(
            {
                "type": "thinking_detail",
                "agent": "Physical Agent",
                "title": "2. 统计学特征比对推演",
                "content": reasoning,
            }
        )

        physical_result = {
            "scan_type": scan_type,
            "physical_metrics": physical_metrics,
            "label": phys_label,
            "reasoning": reasoning,
        }

        new_context = dict(context)
        new_context["physical_metrics"] = physical_metrics
        new_context["physical_diagnosis_label"] = phys_label

        if not new_context.get("image_diagnosis_label") or phys_label == new_context.get("image_diagnosis_label"):
            new_context["last_disease_label"] = phys_label

        return {
            "context": new_context,
            "physical_result": physical_result,
            "progress_event": {"type": "thinking", "msg": f"Node 3B: 数值分析完成，倾向诊断为 {phys_label}"},
            "thinking_details": details,
        }
    except Exception as exc:
        details.append(
            {
                "type": "thinking_detail",
                "agent": "Physical Agent",
                "title": "⚠️ 推理解析崩溃",
                "content": f"模型输出了无法解析的数据格式，导致节点异常中止。\n错误详情：{exc}",
            }
        )
        return {
            "context": {**context, "physical_metrics": physical_metrics},
            "physical_result": {"scan_type": scan_type, "physical_metrics": physical_metrics},
            "error": f"数值节点异常：{exc}",
            "progress_event": progress,
            "thinking_details": details,
        }
