import json
from typing import Dict, Any

from state import AgentState
from config import SYSTEM_PROMPT
from utils import call_llm, emit_stream_event, keep_recent_history


REPORT_KEYWORDS = ["建议", "报告", "方案", "治疗", "用药", "随访", "处置", "下一步", "诊断建议", "诊断报告"]
EXPLANATION_KEYWORDS = [
    "什么是", "是什么意思", "了解", "介绍", "科普", "解释一下", "讲讲", "区别", "严重吗", "危险吗",
    "正常吗", "会不会", "是不是", "为何", "为什么", "原因", "表现", "症状", "会怎样", "怎么看",
]
DISEASE_TERMS = [
    "dr", "amd", "cnv", "csc", "rvo", "normal", "others", "other",
    "糖网", "糖尿病视网膜病变", "黄斑变性", "脉络膜新生血管", "中心性浆液性脉络膜视网膜病变",
    "视网膜静脉阻塞", "正常", "其他", "其他病变",
]

DISEASE_QUERY_PATTERNS = [
    ("DR", ["dr", "糖网", "糖尿病视网膜病变"]),
    ("AMD", ["amd", "黄斑变性", "年龄相关性黄斑变性"]),
    ("CNV", ["cnv", "脉络膜新生血管"]),
    ("CSC", ["csc", "中心性浆液性脉络膜视网膜病变", "中浆"]),
    ("RVO", ["rvo", "视网膜静脉阻塞"]),
    ("NORMAL", ["normal", "正常"]),
    ("OTHERS", ["others", "other", "其他", "其他病变"]),
]
GENERAL_HEALTH_KEYWORDS = [
    "眼部健康", "护眼", "保养", "日常护理", "日常保健", "预防", "怎么预防", "怎么保护",
    "如何保护", "如何保持", "生活习惯", "饮食", "作息", "用眼习惯", "视疲劳", "干眼",
    "复查频率", "多久复查", "平时注意", "注意事项", "健康建议",
]


def wants_explanatory_output(state: AgentState) -> bool:
    user_text = (state.get("user_text", "") or "").lower()
    if not user_text:
        return False
    has_explanation_signal = any(k in user_text for k in EXPLANATION_KEYWORDS)
    mentions_disease = any(term in user_text for term in DISEASE_TERMS)
    return has_explanation_signal and (mentions_disease or state.get("user_intent") == "consult")


def wants_general_health_output(state: AgentState) -> bool:
    user_text = (state.get("user_text", "") or "").lower()
    if not user_text:
        return False
    return any(keyword in user_text for keyword in GENERAL_HEALTH_KEYWORDS)


def detect_queried_disease(user_text: str) -> str | None:
    text = (user_text or "").strip().lower()
    if not text:
        return None
    for canonical, patterns in DISEASE_QUERY_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return canonical
    return None


def build_disease_explanation(disease: str) -> str:
    mapping = {
        "DR": "DR 一般指糖尿病视网膜病变，是糖尿病长期影响视网膜微小血管后出现的一类眼底病变。早期可能没有明显症状，进展后可能出现视力波动、视物模糊、飞蚊增多，严重时会影响视力。",
        "AMD": "AMD 一般指年龄相关性黄斑变性，主要影响黄斑区，也就是负责中心视力和精细视物的部位。常见表现包括中心视力模糊、看直线变弯、阅读吃力，部分患者会逐渐影响看东西的清晰度。",
        "CNV": "CNV 一般指脉络膜新生血管，意思是眼底黄斑附近长出了异常的新生血管。这些血管容易渗漏或出血，所以常见表现是视力突然下降、看东西变形、中央黑影或扭曲感。",
        "CSC": "CSC 一般指中心性浆液性脉络膜视网膜病变，常见于工作压力大、作息紧张的人群。典型表现是单眼看东西像隔着一层水雾、颜色发暗发黄、物体变小或变远，很多患者会先感觉中心视力发虚。",
        "RVO": "RVO 一般指视网膜静脉阻塞，可以理解为眼底静脉回流受阻后引起的一类血管性病变。常见表现是无痛性视力下降、局部视野被遮挡，部分患者会突然感觉看东西少了一块或变暗。",
        "NORMAL": "NORMAL 在这里表示本次分析结果没有发现明确支持特定异常病变的证据，也就是当前更偏向于未见明显异常。",
        "OTHERS": "OTHERS 在这里表示当前结果更接近未被系统细分类标签单独覆盖的其他类型异常，或者属于非典型表现，通常意味着仍需要结合更详细的影像、症状和专科检查进一步判断。",
    }
    return mapping.get(disease, f"{disease} 是一种需要结合具体眼底表现来判断的疾病类型。")


def build_followup_explanation_response(state: AgentState) -> str:
    context = state.get("context", {})
    user_text = state.get("user_text", "") or ""
    asked_disease = detect_queried_disease(user_text)
    final_label = context.get("last_disease_label")
    pathology_result = context.get("pathology_diagnosis_result") or {}
    img_label = context.get("image_diagnosis_label")
    phys_label = context.get("physical_diagnosis_label")
    rag_result = state.get("rag_result") or ""
    warnings = state.get("warnings", []) or []
    warning_text = f"\n\n流程降级信息：{warnings[-1]}" if warnings else ""

    current_result = final_label or img_label or phys_label or "尚未形成稳定结论"
    target_disease = asked_disease or current_result
    explanation = build_disease_explanation(str(target_disease))

    relation_line = ""
    if final_label:
        if asked_disease and asked_disease == final_label:
            relation_line = f"结合你上一轮已经完成的综合分析与裁决，目前最终更偏向于 **{final_label}**。"
        elif asked_disease and asked_disease != final_label:
            relation_line = (
                f"结合你上一轮已经完成的综合分析与裁决，目前最终更偏向于 **{final_label}**，"
                f"并不是 **{asked_disease}**。"
            )
        else:
            relation_line = f"结合你上一轮已经完成的综合分析与裁决，目前最终更偏向于 **{final_label}**。"
    else:
        relation_line = f"结合目前流程中的已有结果，当前更偏向于 **{current_result}**。"

    conflict_line = ""
    if pathology_result and img_label and final_label and img_label != final_label:
        conflict_line = (
            f"需要说明的是，单纯图像初步结果一度更偏向 **{img_label}**，"
            f"但后续结合数值分析和症状会诊后，最终采纳的是 **{final_label}**。"
        )

    tail = f"\n补充参考：{rag_result}" if rag_result else ""
    return "\n".join([part for part in [explanation, relation_line, conflict_line, tail.strip(), warning_text.strip()] if part]).strip()


def wants_report_output(state: AgentState) -> bool:
    user_text = state.get("user_text", "") or ""
    if wants_explanatory_output(state):
        return False
    if wants_general_health_output(state):
        return False
    if state.get("user_intent") == "consult":
        return True
    if state.get("needs_rag"):
        return True
    return any(k in user_text for k in REPORT_KEYWORDS)


def render_non_llm_report(state: AgentState) -> str:
    context = state.get("context", {})
    warnings = state.get("warnings", []) or []
    warning_text = f"\n\n流程降级信息：{warnings[-1]}" if warnings else ""

    img_label = context.get("image_diagnosis_label")
    phys_label = context.get("physical_diagnosis_label")
    pathology_result = context.get("pathology_diagnosis_result")
    diagnosis = context.get("last_disease_label")
    rag_result = state.get("rag_result") or ""

    has_physical = bool(context.get("physical_metrics"))
    has_pathology = bool(pathology_result)
    has_rag = bool(rag_result)

    basis = []
    if img_label:
        basis.append(f"图值专家初步判断：{img_label}")
    if phys_label:
        basis.append(f"数值专家初步判断：{phys_label}")
    if pathology_result:
        basis.append(f"病理裁决倾向：{pathology_result.get('top_3_diagnoses', [diagnosis])[0]}")
    if context.get("metrics"):
        basis.append("已完成视觉分割与关键指标提取")
    if has_physical:
        basis.append("已完成物理指标换算")
    if has_rag:
        basis.append("已完成知识库建议整合")

    if img_label and phys_label and img_label != phys_label and not has_pathology:
        diagnosis_line = f"当前图值诊断倾向于 {img_label}，数值诊断倾向于 {phys_label}，两者存在分歧。"
        next_step = "建议补充症状描述，或进一步完善 OCTA / 眼底相关检查后再进行病理裁决。"
    else:
        diagnosis_line = f"当前综合判断优先考虑：{diagnosis or img_label or phys_label or '尚未形成稳定结论'}。"
        next_step = "建议结合线下专科检查结果进一步确认。"

    report = [
        "一、初步诊断结论",
        diagnosis_line,
        "",
        "二、诊断依据",
        "；".join(basis) if basis else "当前可用依据不足。",
        "",
        "三、已完成的分析环节",
        f"- 图值诊断：{'已完成' if img_label else '未完成'}",
        f"- 物理指标计算：{'已完成' if has_physical else '未完成'}",
        f"- 病理裁决：{'已完成' if has_pathology else '未完成'}",
        f"- RAG建议整合：{'已完成' if has_rag else '未完成'}",
        "",
        "四、风险提示",
        "当前结果属于辅助分析意见，不能替代线下专科医生的最终诊断。",
        "",
        "五、下一步检查建议",
        next_step,
        "",
        "六、诊疗建议",
    ]

    if has_rag:
        report.append(rag_result)
    else:
        report.append("当前未获得足够的知识库治疗建议支撑，因此暂不提供具体用药或治疗方案；建议先明确诊断后，再由眼科专科医生结合病程与检查结果制定处置方案。")

    report.extend(
        [
            "",
            "七、说明与免责",
            "本报告为基于当前上传信息生成的辅助性诊断建议，仅供参考，请以线下医疗机构检查与医生面诊意见为准。",
        ]
    )

    if warning_text:
        report.append(warning_text)

    return "\n".join(report)


def render_non_llm_final(state: AgentState) -> str:
    if wants_report_output(state):
        return render_non_llm_report(state)

    warnings = state.get("warnings", []) or []
    warning_text = f"\n\n流程降级信息：{warnings[-1]}" if warnings else ""

    if state.get("error"):
        return f"本次流程未能完成：{state['error']}" + warning_text
    if state.get("blocked"):
        return "图像已被系统判定为非标准 OCTA 或质量异常，因此本次停止后续诊断。建议重新上传符合要求的图像。" + warning_text
    if state.get("user_intent") == "segment_only":
        metrics = state.get("context", {}).get("metrics")
        physical_metrics = state.get("context", {}).get("physical_metrics")
        return (
            f"已完成图像分割与指标提取。本次按你的要求只返回分割/指标信息，不做疾病诊断。"
            f"\n视觉指标：{metrics}"
            f"\n物理指标：{physical_metrics}" + warning_text
        )

    context = state.get("context", {})
    diagnosis = context.get("last_disease_label")
    pathology_result = context.get("pathology_diagnosis_result")
    rag_result = state.get("rag_result") or ""

    if wants_explanatory_output(state):
        user_text = state.get("user_text", "") or ""
        current_result = diagnosis or context.get("image_diagnosis_label") or context.get("physical_diagnosis_label") or "尚未形成稳定结论"
        asked_disease = detect_queried_disease(user_text)
        if diagnosis or pathology_result:
            return build_followup_explanation_response(state)
        if asked_disease:
            return (
                build_disease_explanation(asked_disease)
                + f"\n结合你这一次的分析结果，目前系统给出的结论更偏向于：{current_result}。"
                + f"\n如果你问的是 {asked_disease}，而当前结果并不是它，这表示本次分析里没有看到明确支持 {asked_disease} 的异常证据；"
                + "但这并不等于以后永远不会发生，仍需要结合症状、病史和定期复查综合判断。"
                + (f"\n另外，知识库补充信息如下：{rag_result}" if rag_result else "")
                + warning_text
            )
        return (
            f"你这次更像是在追问结果含义或疾病知识。当前系统给出的结果偏向于：{current_result}。"
            "如果你愿意，我可以继续用更通俗的方式解释这个结果代表什么、常见症状是什么、是否需要复查或进一步检查。"
            + (f"\n补充参考：{rag_result}" if rag_result else "")
            + warning_text
        )

    if wants_general_health_output(state):
        current_result = diagnosis or context.get("image_diagnosis_label") or context.get("physical_diagnosis_label") or "尚未形成稳定结论"
        return (
            "保持眼部健康通常要从几个方面入手：规律作息、避免长时间连续近距离用眼、控制血糖血压等慢病风险、保持均衡饮食、按需复查。"
            "如果长时间看屏幕，建议有意识休息、增加眨眼、避免熬夜，并在出现视力下降、变形、黑影遮挡等情况时尽快检查。"
            + f"\n结合你目前这次的分析结果，当前结论更偏向于：{current_result}。"
            + (f"\n补充参考：{rag_result}" if rag_result else "")
            + warning_text
        )

    if pathology_result:
        return (
            f"当前综合会诊后，优先考虑：{diagnosis}。"
            f"\n病理裁决依据：{pathology_result.get('reasoning', '无')}"
            + (f"\n参考建议：{rag_result}" if rag_result else "")
            + warning_text
        )

    if diagnosis and state.get("needs_rag") and rag_result:
        return f"初步判断为：{diagnosis}。以下为结合知识库整理的建议：\n{rag_result}" + warning_text
    if diagnosis:
        return f"初步判断为：{diagnosis}。本轮未检索治疗方案，因为你没有明确要求方案/用药建议。" + warning_text

    if warnings:
        return f"主流程存在异常，但系统已尝试自动降级。当前未形成稳定诊断结论。{warning_text}"
    return "已完成本轮处理。"


def build_report_prompt(summary_payload: Dict[str, Any]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "你现在不是在自由总结，而是在生成【眼底影像诊断建议报告】。\n"
        "请严格按以下结构输出，不要省略标题，不要输出 JSON，不要写代码块：\n\n"
        "一、初步诊断结论\n"
        "二、诊断依据\n"
        "三、已完成的分析环节\n"
        "四、风险提示\n"
        "五、下一步检查建议\n"
        "六、诊疗建议\n"
        "七、说明与免责\n\n"
        "要求：\n"
        "1. 如果没有 RAG 结果，不要编造用药/治疗方案，只能给出保守的下一步检查或就诊建议。\n"
        "2. 如果 physical/pathology 未完成，要明确写“未完成”，不要假装完成。\n"
        "3. 如果图值与数值有分歧，但尚未病理裁决，只能输出“存在分歧，建议补充症状或进一步检查”。\n"
        "4. 只有当 rag_result 有内容时，才能输出更具体的诊疗建议。\n"
        "5. 建议必须具体，不能只写“请咨询医生”这一句。\n"
        "6. 如果用户明确要求的是报告，请使用正式、结构化、像临床辅助报告的中文风格。\n\n"
        f"当前结构化输入如下：\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
    )


def build_summary_prompt(summary_payload: Dict[str, Any]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "以下是结构化工作流输出（包含图值、数值、病理多路会诊结果），请基于这些结果回答用户。\n"
        "要求：\n"
        "1. 清楚说明最终倾向诊断。\n"
        "2. 明确说明是否完成图值诊断、物理指标计算、病理裁决、RAG建议整合。\n"
        "3. 若有流程降级或失败，请自然说明。\n"
        "4. 不要输出 JSON，不要写代码块。\n\n"
        f"{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
    )


def build_explanatory_prompt(summary_payload: Dict[str, Any]) -> str:
    asked_disease = detect_queried_disease(str(summary_payload.get("用户本轮问题") or ""))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "用户当前是在做【解释型咨询 / 日常追问】，不是要正式报告。\n"
        "请直接、自然、像医生在门诊解释一样回答。\n"
        "回答要求：\n"
        "1. 先正面回答用户的问题，例如“什么是DR”“这是否正常”“为什么会这样”。\n"
        "2. 如果用户提到了某个疾病缩写或病名，先用通俗语言解释该疾病是什么、常见表现是什么，不要被当前诊断结果带偏话题。\n"
        "3. 如果本次工作流已经得出结果，要明确说明“这次结果”和用户追问的疾病/概念之间是什么关系。\n"
        "4. 如果本次结果是 NORMAL，要明确告诉用户“目前这次分析没有支持该病的异常证据”，但不要因此回避解释该病本身。\n"
        "5. 如果有 rag_result，可以整合为简洁的医学知识补充；如果没有，也要基于现有结构化结果给出可理解解释。\n"
        "6. 不要写成条块化正式报告；优先用自然中文分段说明。\n"
        "7. 不要输出 JSON，不要写代码块。\n\n"
        f"用户当前明确追问的疾病/主题：{asked_disease or '未显式识别'}\n\n"
        f"当前结构化输入如下：\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
    )


def build_general_health_prompt(summary_payload: Dict[str, Any]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "用户当前是在做【日常护眼 / 泛健康咨询】，不是在追问具体病变定义，也不是要正式报告。\n"
        "请直接、自然、实用地回答。\n"
        "回答要求：\n"
        "1. 直接回答用户关于护眼、生活方式、饮食、复查、预防的问题。\n"
        "2. 如果本次已有工作流结果，可简洁说明这次结果和日常建议的关系。\n"
        "3. 建议以日常可执行事项为主，不要空泛。\n"
        "4. 不要写成正式医疗报告，不要输出 JSON 或代码块。\n\n"
        f"当前结构化输入如下：\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
    )


async def final_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context", {})
    img_label = context.get("image_diagnosis_label")
    phys_label = context.get("physical_diagnosis_label")
    pathology_result = context.get("pathology_diagnosis_result")

    # 图值 / 数值分歧，且尚未病理裁决 -> 先追问症状
    if img_label and phys_label and img_label != phys_label and not pathology_result and not context.get("symptoms_requested"):
        question = (
            f"图值分析系统提示为 **{img_label}**，但体检数值交叉比对分析倾向于 **{phys_label}**。两者存在分歧。\n\n"
            "为了让我为你进行更准确的病理裁决，请问你最近眼睛有以下哪些不适？\n"
            "- 是有中心视力模糊、看直线弯曲？\n"
            "- 是有突发性的视力断崖式下降或黑影遮挡？\n"
            "- 还是像隔着水雾，看东西变暗发黄？\n"
            "- 或者是飞蚊症突然加重？\n"
            "（请用你自己的话简单描述一下你的感受）"
        )
        new_context = dict(context)
        new_context["symptoms_requested"] = True
        new_context["pending_report_after_pathology"] = bool(wants_report_output(state) or state.get("needs_rag"))

        new_history = list(state.get("history", []))
        new_history.append({"role": "assistant", "content": question})
        return {"final_text": question, "context": new_context, "history": new_history}

    progress = {"type": "thinking", "msg": "Node 5: 正在整合各专家会诊结果并生成最终答复..."}
    emit_stream_event(progress)

    has_physical = bool(context.get("physical_metrics"))
    has_pathology = bool(pathology_result)
    has_rag = bool(state.get("rag_result"))
    report_mode = wants_report_output(state)
    explanatory_mode = wants_explanatory_output(state)
    general_health_mode = wants_general_health_output(state)

    summary_payload = {
        "用户本轮问题": state.get("user_text"),
        "工作流意图": state.get("user_intent"),
        "视觉指标": context.get("metrics"),
        "物理指标": context.get("physical_metrics"),
        "图值专家初步诊断": img_label,
        "数值专家初步诊断": phys_label,
        "病理专家最终裁决": pathology_result,
        "最终采纳疾病标签": context.get("last_disease_label"),
        "RAG文献": state.get("rag_result"),
        "是否完成图值诊断": bool(img_label),
        "是否完成物理分析": has_physical,
        "是否完成病理裁决": has_pathology,
        "是否完成RAG建议整合": has_rag,
        "agent_status": state.get("agent_status"),
        "warnings": state.get("warnings"),
        "blocked": state.get("blocked", False),
        "error": state.get("error"),
    }

    if explanatory_mode:
        if context.get("last_disease_label") or pathology_result:
            final_text = build_followup_explanation_response(state)
            new_history = list(state.get("history", []))
            new_history.append({"role": "assistant", "content": final_text})
            new_history = keep_recent_history(new_history)
            return {
                "final_text": final_text,
                "history": new_history,
                "context": {**context, "pending_report_after_pathology": False},
                "progress_event": progress,
            }
        prompt = build_explanatory_prompt(summary_payload)
    elif general_health_mode:
        prompt = build_general_health_prompt(summary_payload)
    else:
        prompt = build_report_prompt(summary_payload) if report_mode else build_summary_prompt(summary_payload)

    try:
        final_text = await call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
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
        "context": {**context, "pending_report_after_pathology": False},
        "progress_event": progress,
    }
