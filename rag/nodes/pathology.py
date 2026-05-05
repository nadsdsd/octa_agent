import json
from typing import Dict, Any

from state import AgentState
from config import DISEASE_SYMPTOMS, PATHOLOGY_LLM_PROVIDER
from utils import call_llm, emit_stream_event, extract_json_object


async def pathology_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context", {})
    user_text = state.get("user_text", "")

    img_label = context.get("image_diagnosis_label", "未知")
    image_confidence = context.get("image_confidence")
    phys_label = context.get("physical_diagnosis_label", "未知")
    physical_metrics = context.get("physical_metrics") or {}
    physical_confidence = context.get("physical_confidence")

    progress = {
        "type": "thinking",
        "msg": "Node 4: 唤醒 Pathology Agent，正在结合患者症状进行最终会诊..."
    }
    emit_stream_event(progress)

    details = []

    # 如果两个标签本来就一致，其实不需要再让模型自由发挥
    if img_label == phys_label and img_label != "未知":
        reasoning = (
            f"图值分析师与数值分析师结论一致，均指向 {img_label}。"
            f"{f' 图像分类模型置信度约为 {float(image_confidence):.2f}。' if image_confidence is not None else ''}"
            f"{f' 数值分析师辅助置信度约为 {float(physical_confidence):.2f}。' if physical_confidence is not None else ''}"
            "未出现标签冲突，因此直接采纳一致结论。"
        )
        pathology_result = {
            "reasoning": reasoning,
            "final_choice": img_label,
            "candidate_labels": [img_label, phys_label],
            "top_3_diagnoses": [img_label, phys_label],
        }

        new_context = dict(context)
        new_context["pathology_diagnosis_result"] = pathology_result
        new_context["last_disease_label"] = img_label
        new_context["symptoms_requested"] = False

        details.append(
            {
                "type": "thinking_detail",
                "agent": "Pathology Agent",
                "title": "会诊结果",
                "content": reasoning,
            }
        )

        return {
            "context": new_context,
            "pathology_result": pathology_result,
            "progress_event": {
                "type": "thinking",
                "msg": f"Node 4: 专家会诊完毕，首选诊断为 {img_label}"
            },
            "thinking_details": details,
        }

    prompt = (
        "你是权威的眼底病理诊断专家。现在需要你在两个候选诊断中做最终裁决。\n\n"
        f"候选诊断 A（图值分析师，基于图像神经网络）: {img_label}\n"
        f"候选诊断 B（数值分析师，基于指标统计学特征）: {phys_label}\n"
        f"图值分析师当前分类置信度: {f'{float(image_confidence):.3f}' if image_confidence is not None else '未提供'}\n"
        f"数值分析师当前辅助置信度: {f'{float(physical_confidence):.3f}' if physical_confidence is not None else '未提供显式概率，仅可视为中等强度辅助证据'}\n"
        f"患者自述的近期眼部症状: {user_text or '未提供明确症状'}\n"
        f"患者关键物理指标: {json.dumps(physical_metrics, ensure_ascii=False)}\n\n"
        f"可参考的眼底疾病临床症状学知识:\n{DISEASE_SYMPTOMS}\n\n"
        "你的任务要求如下：\n"
        "1. 你只能在候选诊断 A 和候选诊断 B 之间二选一，不能新增第三种疾病，不能输出候选集合之外的标签。\n"
        "2. 你必须综合考虑：症状匹配度、图像分类置信度、以及数值分析结论是否只是辅助证据。\n"
        "3. 如果图像分类置信度较低（例如接近 0.5）且症状与数值分析师更一致，不要机械地偏向候选 A。\n"
        "4. 如果数值分析师没有显式概率，则只能把它视为辅助证据，不能因为它存在就压过更强的图像和症状证据。\n"
        "5. 若症状信息不足或两者证据接近，优先选择更保守、且证据链更完整的一方，并说明理由。\n"
        "6. 输出必须是合法 JSON，且字符串中不要出现真实换行。\n\n"
        "输出格式：\n"
        "{\n"
        f'  "candidate_labels": ["{img_label}", "{phys_label}"],\n'
        '  "reasoning": "简要说明为什么在两个候选标签中选择其中一个，并说明你如何综合图像分类置信度、症状和数值辅助证据",\n'
        f'  "final_choice": "只能是 {img_label} 或 {phys_label}"\n'
        "}"
    )

    try:
        response = await call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            mode=PATHOLOGY_LLM_PROVIDER,
            agent_name="pathology",
        )
        parsed = extract_json_object(response)

        reasoning = parsed.get("reasoning", "无详细推理过程。")
        final_choice = parsed.get("final_choice", img_label)

        # 强约束兜底：即使模型乱输出，也只能在前两个标签里选
        if final_choice not in {img_label, phys_label}:
            final_choice = img_label

        pathology_result = {
            "candidate_labels": [img_label, phys_label],
            "reasoning": reasoning,
            "final_choice": final_choice,
            # 为了兼容你现有下游逻辑，继续保留 top_3_diagnoses 这个字段
            # 但这里只放两个候选，且首位一定是最终选择
            "top_3_diagnoses": [
                final_choice,
                phys_label if final_choice == img_label else img_label,
            ],
        }

        details.append(
            {
                "type": "thinking_detail",
                "agent": "Pathology Agent",
                "title": "症状与数据综合研判",
                "content": reasoning,
            }
        )

        new_context = dict(context)
        new_context["pathology_diagnosis_result"] = pathology_result
        new_context["last_disease_label"] = final_choice
        new_context["symptoms_requested"] = False

        return {
            "context": new_context,
            "pathology_result": pathology_result,
            "progress_event": {
                "type": "thinking",
                "msg": f"Node 4: 专家会诊完毕，首选诊断为 {new_context['last_disease_label']}"
            },
            "thinking_details": details,
        }

    except Exception as exc:
        details.append(
            {
                "type": "thinking_detail",
                "agent": "Pathology Agent",
                "title": "⚠️ 会诊解析崩溃",
                "content": f"模型输出了无法解析的数据格式，导致节点异常中止。错误详情：{exc}",
            }
        )
        return {
            "error": f"病理节点异常：{exc}",
            "progress_event": progress,
            "thinking_details": details,
        }
