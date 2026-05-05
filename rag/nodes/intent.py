from typing import Dict, Any

from state import AgentState
from utils import call_llm, extract_json_object, emit_stream_event


STRONG_CONSULT_KEYWORDS = [
    "建议", "报告", "方案", "治疗", "用药", "随访", "处置", "下一步",
    "诊断建议", "诊疗建议", "诊断报告", "治疗建议", "进一步建议", "进一步报告",
]

SEGMENT_KEYWORDS = ["分割", "mask", "rv", "faz", "血流指标", "指标", "只看分割", "仅分割"]
CONSULT_KEYWORDS = ["治疗", "方案", "怎么治", "用药", "报告", "建议", "咨询", "指南", "医学知识", "随访", "处置"]
DIAGNOSIS_KEYWORDS = ["诊断", "分析", "判断", "病变", "筛查", "看看这张图", "帮我看", "识别"]
EXPLANATION_KEYWORDS = [
    "什么是", "是什么意思", "了解", "介绍", "科普", "解释一下", "讲讲", "区别", "严重吗", "危险吗",
    "正常吗", "会不会", "是不是", "为何", "为什么", "原因", "表现", "症状", "会怎样", "怎么看",
]
DISEASE_TERMS = [
    "dr", "amd", "cnv", "csc", "rvo", "normal", "others", "other",
    "糖网", "糖尿病视网膜病变", "黄斑变性", "脉络膜新生血管", "中心性浆液性脉络膜视网膜病变",
    "视网膜静脉阻塞", "正常", "其他", "其他病变",
]
GENERAL_HEALTH_KEYWORDS = [
    "眼部健康", "护眼", "保养", "日常护理", "日常保健", "预防", "怎么预防", "怎么保护",
    "如何保护", "如何保持", "生活习惯", "饮食", "作息", "用眼习惯", "视疲劳", "干眼",
    "复查频率", "多久复查", "平时注意", "注意事项", "健康建议",
]


def is_informational_consult(user_text: str, has_prior_dx: bool) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False

    has_explanation_signal = any(k in text for k in EXPLANATION_KEYWORDS)
    mentions_disease = any(term in text for term in DISEASE_TERMS)

    if has_explanation_signal and mentions_disease:
        return True
    if has_prior_dx and has_explanation_signal:
        return True
    return False


def is_general_eye_health_consult(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in GENERAL_HEALTH_KEYWORDS)


def heuristic_intent_fallback(user_text: str, has_image: bool, has_prior_dx: bool) -> Dict[str, Any]:
    text = (user_text or "").lower()

    wants_segment = any(k in text for k in SEGMENT_KEYWORDS)
    wants_consult = any(k in text for k in CONSULT_KEYWORDS)
    wants_diagnosis = any(k in text for k in DIAGNOSIS_KEYWORDS)
    wants_explanation = is_informational_consult(user_text, has_prior_dx)
    wants_general_health = is_general_eye_health_consult(user_text)

    if has_image and wants_segment and not wants_consult and not wants_diagnosis:
        return {
            "user_intent": "segment_only",
            "needs_vision": True,
            "needs_diagnosis": False,
            "needs_physical": False,
            "needs_pathology": False,
            "needs_rag": False,
            "route_reason": "命中分割类关键词，且未要求诊断/方案。",
        }

    if wants_consult or wants_explanation or wants_general_health:
        return {
            "user_intent": "consult",
            "needs_vision": bool(has_image and not has_prior_dx),
            "needs_diagnosis": bool(has_image and not has_prior_dx),
            "needs_physical": bool(has_image and not has_prior_dx),
            "needs_pathology": False,
            "needs_rag": True,
            "route_reason": "命中治疗建议、解释性咨询或日常护眼咨询关键词。",
        }

    if has_image or wants_diagnosis:
        return {
            "user_intent": "diagnosis",
            "needs_vision": bool(has_image),
            "needs_diagnosis": True,
            "needs_physical": True,
            "needs_pathology": False,
            "needs_rag": False,
            "route_reason": "默认进入图像分析/诊断路径。",
        }

    return {
        "user_intent": "general_chat",
        "needs_vision": False,
        "needs_diagnosis": False,
        "needs_physical": False,
        "needs_pathology": False,
        "needs_rag": wants_consult or wants_explanation or wants_general_health,
        "route_reason": "无图像输入，按普通咨询或闲聊处理。",
    }


def apply_strong_consult_override(state: AgentState, routing: Dict[str, Any]) -> Dict[str, Any]:
    """
    强规则覆盖：
    只要用户明确要求“建议/报告/方案/治疗”，就强制视为 consult，
    并确保 needs_rag=True。
    """
    user_text = state.get("user_text", "") or ""
    context = state.get("context", {}) or {}

    if not any(k in user_text for k in STRONG_CONSULT_KEYWORDS):
        if not is_informational_consult(user_text, bool(context.get("last_disease_label"))) and not is_general_eye_health_consult(user_text):
            return routing

    has_image = bool(state.get("image_path"))
    has_dx = bool(context.get("last_disease_label"))
    has_metrics = bool(context.get("metrics"))

    routing["user_intent"] = "consult"
    routing["needs_rag"] = True
    routing["needs_pathology"] = False
    routing["route_reason"] = "命中强规则：用户在追问诊断解释、医学知识、日常护眼或诊疗建议。"

    # 如果已经有稳定诊断，直接走 RAG/Final，不重复跑图像链路
    if has_dx:
        routing["needs_vision"] = False
        routing["needs_diagnosis"] = False
        routing["needs_physical"] = False
        return routing

    # 如果没有稳定诊断，但这轮有图或已有视觉特征，就补齐分析链路
    if has_image or has_metrics:
        routing["needs_vision"] = bool(has_image and not has_metrics)
        routing["needs_diagnosis"] = True
        routing["needs_physical"] = True
        return routing

    # 没图也没历史诊断，只能先走 consult 文本答复，但保留 RAG
    routing["needs_vision"] = False
    routing["needs_diagnosis"] = False
    routing["needs_physical"] = False
    return routing


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
3. physical（数值指标会诊）
4. pathology（分歧后病理裁决）
5. rag（医学知识检索）

规则：
- 如果用户只要求看分割、mask、FAZ/RV、指标，intent=segment_only，needs_vision=true，needs_diagnosis=false，needs_physical=false，needs_rag=false。
- 如果用户上传了图像并要求“分析/诊断/看看这张图”，intent=diagnosis，needs_vision=true，needs_diagnosis=true，needs_physical=true。
- 如果用户明确要求“建议/报告/方案/治疗/用药/随访/下一步”，intent=consult，needs_rag=true。
- 如果用户在追问“什么是DR / 这正常吗 / 为什么这样 / 严重吗 / 想了解某个病”，这属于解释型咨询，intent=consult，needs_rag=true。
- 如果用户在问“如何保持眼部健康 / 如何护眼 / 平时要注意什么 / 多久复查 / 如何预防”，这属于泛健康咨询，intent=consult，needs_rag=true。
- 如果当前没有新图像，但历史中已经有诊断，而用户本轮是在追问治疗/建议/报告，则 needs_vision=false，needs_diagnosis=false，needs_physical=false，needs_rag=true。
- 如果当前没有新图像，但历史中已经有诊断，而用户本轮是在追问病名含义、疾病知识、正常与否、风险或原因，也应 needs_vision=false，needs_diagnosis=false，needs_physical=false，needs_rag=true。
- 如果当前没有新图像，但用户本轮是在问日常护眼、复查频率、生活方式、预防建议，也应 needs_vision=false，needs_diagnosis=false，needs_physical=false，needs_rag=true。
- pathology 只有在用户正在补充症状时才置为 true。
- 如果没有图像且只是普通闲聊，intent=general_chat。

返回格式：
{{
  "user_intent": "segment_only | diagnosis | consult | general_chat",
  "needs_vision": true,
  "needs_diagnosis": false,
  "needs_physical": false,
  "needs_pathology": false,
  "needs_rag": false,
  "route_reason": "一句中文解释"
}}

输入信息：
- 用户文本: {user_text or '<空>'}
- 是否上传图像: {str(has_image).lower()}
- 历史是否已有诊断: {str(has_prior_dx).lower()}
- 历史疾病标签: {prior_label or '<无>'}
    """.strip()

    content = await call_llm([{"role": "user", "content": router_prompt}], 0.0)
    parsed = extract_json_object(content)

    validated = {
        "user_intent": parsed.get("user_intent", "diagnosis"),
        "needs_vision": bool(parsed.get("needs_vision", False)),
        "needs_diagnosis": bool(parsed.get("needs_diagnosis", False)),
        "needs_physical": bool(parsed.get("needs_physical", parsed.get("needs_diagnosis", False))),
        "needs_pathology": bool(parsed.get("needs_pathology", False)),
        "needs_rag": bool(parsed.get("needs_rag", False)),
        "route_reason": str(parsed.get("route_reason", "LLM 已完成路由判断。")),
    }

    allowed = {"segment_only", "diagnosis", "consult", "general_chat"}
    if validated["user_intent"] not in allowed:
        raise ValueError(f"非法 intent: {validated['user_intent']}")

    return validated


async def intent_node(state: AgentState) -> Dict[str, Any]:
    context = state.get("context", {})

    # 症状追问场景：直接送病理裁决
    if context.get("symptoms_requested") is True:
        progress = {"type": "thinking", "msg": "Node 1: 收到用户症状描述，正在转交病理分析师..."}
        emit_stream_event(progress)
        return {
            "user_intent": "provide_symptoms",
            "needs_vision": False,
            "needs_diagnosis": False,
            "needs_physical": False,
            "needs_pathology": True,
            "needs_rag": bool(context.get("pending_report_after_pathology")),
            "route_reason": "用户正在回复分歧裁决所需的症状描述。",
            "progress_event": progress,
        }

    has_image = bool(state.get("image_path"))
    has_prior_dx = bool(context.get("last_disease_label"))

    progress = {"type": "thinking", "msg": "Node 1: 正在识别用户意图并规划工作流..."}
    emit_stream_event(progress)

    update: Dict[str, Any] = {"progress_event": progress}

    try:
        routing = await classify_intent_with_llm(state)
    except Exception:
        routing = heuristic_intent_fallback(state.get("user_text", ""), has_image, has_prior_dx)

    # 兜底补齐
    routing.setdefault("needs_physical", bool(routing.get("needs_diagnosis")))
    routing.setdefault("needs_pathology", False)
    routing.setdefault("needs_rag", False)
    routing.setdefault("route_reason", "启发式路由")

    # 强规则覆盖：用户明确要建议/报告/方案时，必须 consult + rag
    routing = apply_strong_consult_override(state, routing)

    if routing["needs_vision"] and not has_image and not context.get("metrics"):
        update.update(
            {
                **routing,
                "error": "当前没有可分析的图像，也没有历史视觉特征，无法执行分割/诊断。",
                "progress_event": {"type": "thinking", "msg": "Node 1: 已识别到缺少图像输入，准备直接结束。"},
            }
        )
        return update

    if routing["needs_diagnosis"] and not has_image and not context.get("metrics"):
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
