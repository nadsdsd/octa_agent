import asyncio
import json
from typing import Dict, Any, List

from state import AgentState
from config import get_vector_db, DEFAULT_LLM_PROVIDER
from utils import emit_stream_event, call_llm, extract_json_object


GENERAL_HEALTH_KEYWORDS = [
    "眼部健康", "护眼", "保养", "日常护理", "日常保健", "预防", "怎么预防", "怎么保护",
    "如何保护", "如何保持", "生活习惯", "饮食", "作息", "用眼习惯", "视疲劳", "干眼",
    "复查频率", "多久复查", "平时注意", "注意事项", "健康建议",
]


def is_general_eye_health_query(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in GENERAL_HEALTH_KEYWORDS)


def build_rag_query(state: AgentState) -> str:
    user_text = (state.get("user_text") or "").strip()
    disease_label = state.get("context", {}).get("last_disease_label") or ""

    if is_general_eye_health_query(user_text):
        return user_text or "如何保持眼部健康 日常护眼 复查与预防"
    if disease_label and user_text:
        return f"{disease_label} {user_text}"
    if disease_label:
        return f"{disease_label} 治疗方案 指南"
    if user_text:
        return user_text
    return "眼底疾病 治疗方案 指南"


def _pack_doc_texts(docs: List[Any]) -> List[str]:
    return [doc.page_content for doc in docs if getattr(doc, "page_content", "")]


async def grade_retrieved_docs(query: str, doc_texts: List[str]) -> Dict[str, Any]:
    if not doc_texts:
        return {"relevant": False, "reason": "未召回任何有效片段"}

    joined_docs = "\n\n".join(
        [f"片段{i+1}: {text[:800]}" for i, text in enumerate(doc_texts[:3])]
    )

    prompt = (
        "你是RAG检索质量审查员。\n"
        "请判断下面召回片段是否足以支持回答用户问题。\n\n"
        f"用户问题：{query}\n\n"
        f"召回片段：\n{joined_docs}\n\n"
        "判断标准：\n"
        "1. 片段是否和问题主题直接相关；\n"
        "2. 是否包含足够回答问题的知识；\n"
        "3. 如果只是泛泛提到相关疾病，但不能支持该问题，则判为不相关。\n\n"
        "请输出 JSON：\n"
        '{'
        '"relevant": true 或 false, '
        '"reason": "一句话说明原因"'
        '}'
    )

    try:
        response = await call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            mode=DEFAULT_LLM_PROVIDER,
            agent_name="rag_grader",
        )
        parsed = extract_json_object(response)
        return {
            "relevant": bool(parsed.get("relevant", False)),
            "reason": parsed.get("reason", "未提供原因"),
        }
    except Exception as exc:
        return {
            "relevant": False,
            "reason": f"召回审查失败，触发二次检索：{exc}",
        }


async def build_hypothetical_answer(state: AgentState, original_query: str) -> str:
    user_text = (state.get("user_text") or "").strip()
    disease_label = state.get("context", {}).get("last_disease_label") or ""
    physical_metrics = state.get("context", {}).get("physical_metrics") or {}

    prompt = (
        "你是医学知识检索增强器。"
        "请不要回答用户，而是根据用户问题，生成一段“可能出现在权威医学资料中的假设性回答摘要”，"
        "用于辅助向量检索。"
        "\n要求："
        "\n1. 内容要紧扣用户问题；"
        "\n2. 如果已有疾病标签，请围绕该疾病生成；"
        "\n3. 可以补充治疗、诊断、随访、检查、用药、禁忌等相关医学关键词；"
        "\n4. 输出一段自然中文，不要 JSON，不要解释你在做什么。"
        f"\n\n用户原问题：{user_text or original_query}"
        f"\n当前疾病标签：{disease_label or '未知'}"
        f"\n当前关键指标：{json.dumps(physical_metrics, ensure_ascii=False)}"
    )

    try:
        response = await call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            mode=DEFAULT_LLM_PROVIDER,
            agent_name="rag_hyde",
        )
        hyde_text = (response or "").strip()
        if hyde_text:
            return hyde_text
    except Exception:
        pass

    fallback = f"{original_query} {disease_label} 治疗方案 指南 随访 检查 用药 禁忌"
    return fallback.strip()


async def rag_node(state: AgentState) -> Dict[str, Any]:
    query = build_rag_query(state)
    progress = {
        "type": "thinking",
        "msg": f"Node 4R: 唤醒 RAG Agent，正在查阅知识库：{query}"
    }
    emit_stream_event(progress)

    try:
        vector_db = get_vector_db()

        docs = await asyncio.to_thread(vector_db.similarity_search, query, 3)
        doc_texts = _pack_doc_texts(docs)

        if not doc_texts:
            emit_stream_event({
                "type": "thinking_detail",
                "agent": "RAG Agent",
                "title": "初次召回结果",
                "content": "未命中任何有效知识片段。",
            })

            hyde_query = await build_hypothetical_answer(state, query)
            emit_stream_event({
                "type": "thinking_detail",
                "agent": "RAG Agent",
                "title": "二次检索触发",
                "content": f"初次召回为空，改用假设回答增强查询：{hyde_query[:200]}",
            })

            docs_retry = await asyncio.to_thread(vector_db.similarity_search, hyde_query, 3)
            retry_texts = _pack_doc_texts(docs_retry)

            if not retry_texts:
                return {
                    "rag_docs": [],
                    "rag_result": "",
                    "progress_event": {
                        "type": "thinking",
                        "msg": "Node 4R: 两次检索后仍未命中有效文献片段。"
                    },
                }

            rag_result = "\n\n".join(
                [f"【文献参考 {idx + 1}】{text}" for idx, text in enumerate(retry_texts)]
            )
            return {
                "rag_docs": retry_texts,
                "rag_result": rag_result,
                "progress_event": {
                    "type": "thinking",
                    "msg": f"Node 4R: 二次检索完成，共命中 {len(retry_texts)} 条参考片段。"
                },
            }

        grade = await grade_retrieved_docs(query, doc_texts)
        emit_stream_event({
            "type": "thinking_detail",
            "agent": "RAG Agent",
            "title": "召回相关性审查",
            "content": f"相关性判断：{'通过' if grade['relevant'] else '不通过'}；原因：{grade['reason']}",
        })

        if grade["relevant"]:
            rag_result = "\n\n".join(
                [f"【文献参考 {idx + 1}】{text}" for idx, text in enumerate(doc_texts)]
            )
            return {
                "rag_docs": doc_texts,
                "rag_result": rag_result,
                "progress_event": {
                    "type": "thinking",
                    "msg": f"Node 4R: 知识库检索完成，共命中 {len(doc_texts)} 条参考片段。"
                },
            }

        hyde_query = await build_hypothetical_answer(state, query)
        emit_stream_event({
            "type": "thinking_detail",
            "agent": "RAG Agent",
            "title": "二次检索触发",
            "content": f"初次召回相关性不足，使用假设回答增强查询：{hyde_query[:300]}",
        })

        docs_retry = await asyncio.to_thread(vector_db.similarity_search, hyde_query, 3)
        retry_texts = _pack_doc_texts(docs_retry)

        if not retry_texts:
            return {
                "rag_docs": [],
                "rag_result": "",
                "progress_event": {
                    "type": "thinking",
                    "msg": "Node 4R: 二次检索后仍未命中有效文献片段。"
                },
            }

        rag_result = "\n\n".join(
            [f"【文献参考 {idx + 1}】{text}" for idx, text in enumerate(retry_texts)]
        )
        return {
            "rag_docs": retry_texts,
            "rag_result": rag_result,
            "progress_event": {
                "type": "thinking",
                "msg": f"Node 4R: 二次检索完成，共命中 {len(retry_texts)} 条参考片段。"
            },
        }

    except Exception as exc:
        return {
            "error": f"知识检索失败：{exc}",
            "progress_event": progress,
        }
