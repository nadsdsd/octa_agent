import asyncio
from typing import Dict, Any

from state import AgentState
from config import get_vector_db
from utils import emit_stream_event


def build_rag_query(state: AgentState) -> str:
    user_text = (state.get("user_text") or "").strip()
    disease_label = state.get("context", {}).get("last_disease_label") or ""
    if disease_label and user_text:
        return f"{disease_label} {user_text}"
    if disease_label:
        return f"{disease_label} 治疗方案 指南"
    if user_text:
        return user_text
    return "眼底疾病 治疗方案 指南"


async def rag_node(state: AgentState) -> Dict[str, Any]:
    query = build_rag_query(state)
    progress = {"type": "thinking", "msg": f"Node 4R: 唤醒 RAG Agent，正在查阅知识库：{query}"}
    emit_stream_event(progress)

    try:
        vector_db = get_vector_db()
        docs = await asyncio.to_thread(vector_db.similarity_search, query, 3)
        doc_texts = [doc.page_content for doc in docs if getattr(doc, "page_content", "")]
        if not doc_texts:
            return {
                "rag_docs": [],
                "rag_result": "",
                "progress_event": {"type": "thinking", "msg": "Node 4R: 知识库检索完成，但未命中有效文献片段。"},
            }

        rag_result = "\n\n".join([f"【文献参考 {idx + 1}】{text}" for idx, text in enumerate(doc_texts)])
        return {
            "rag_docs": doc_texts,
            "rag_result": rag_result,
            "progress_event": {"type": "thinking", "msg": f"Node 4R: 知识库检索完成，共命中 {len(doc_texts)} 条参考片段。"},
        }
    except Exception as exc:
        return {"error": f"知识检索失败：{exc}", "progress_event": progress}
