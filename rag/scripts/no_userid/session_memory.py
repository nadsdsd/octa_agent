from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Integer, JSON, String

from database import Base, SessionLocal, engine


class SessionMemory(Base):
    __tablename__ = "user_session_memory"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=True, index=True)

    history_json = Column(JSON, nullable=False, default=list)
    context_json = Column(JSON, nullable=False, default=dict)
    diagnosis_history_json = Column(JSON, nullable=False, default=list)
    last_disease_label = Column(String(64), nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# 只把“恢复会话真正需要”的结构化字段落库，不把大体积 mask/base64 落库
PERSISTENT_CONTEXT_KEYS = {
    "metrics",
    "scan_type",
    "diagnosis_result",
    "last_disease_label",
    "physical_metrics",
    "image_diagnosis_label",
    "physical_diagnosis_label",
    "pathology_diagnosis_result",
    "symptoms_requested",
}


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    return str(value)


def _compact_context_for_db(context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    compact = {k: copy.deepcopy(v) for k, v in context.items() if k in PERSISTENT_CONTEXT_KEYS}
    return _safe_jsonable(compact)


def _build_diagnosis_history_entry(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(context, dict):
        return None

    image_label = context.get("image_diagnosis_label")
    physical_label = context.get("physical_diagnosis_label")
    pathology_result = context.get("pathology_diagnosis_result") or {}
    final_label = context.get("last_disease_label")

    if not any([image_label, physical_label, pathology_result, final_label]):
        return None

    pathology_top1 = None
    if isinstance(pathology_result, dict):
        top3 = pathology_result.get("top_3_diagnoses") or []
        if top3:
            pathology_top1 = top3[0]

    return {
        "ts": datetime.utcnow().isoformat(),
        "image_diagnosis_label": image_label,
        "physical_diagnosis_label": physical_label,
        "pathology_top1": pathology_top1,
        "last_disease_label": final_label,
    }


class SessionMemoryStore:
    def __init__(self):
        Base.metadata.create_all(bind=engine)

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None

        db = SessionLocal()
        try:
            row = db.query(SessionMemory).filter(SessionMemory.session_id == session_id).first()
            if not row:
                return None
            return {
                "history": row.history_json or [],
                "context": row.context_json or {},
            }
        finally:
            db.close()

    def save(self, session_id: str, state: Dict[str, Any], user_id: Optional[int] = None) -> None:
        if not session_id:
            return

        history = _safe_jsonable(state.get("history") or [])
        context = _compact_context_for_db(state.get("context") or {})
        last_disease_label = None
        if isinstance(context, dict):
            last_disease_label = context.get("last_disease_label")
        diag_entry = _build_diagnosis_history_entry(context)

        db = SessionLocal()
        try:
            row = db.query(SessionMemory).filter(SessionMemory.session_id == session_id).first()
            if row is None:
                diagnosis_history = [diag_entry] if diag_entry else []
                row = SessionMemory(
                    session_id=session_id,
                    user_id=user_id,
                    history_json=history,
                    context_json=context,
                    diagnosis_history_json=diagnosis_history,
                    last_disease_label=last_disease_label,
                )
                db.add(row)
            else:
                row.history_json = history
                row.context_json = context
                row.last_disease_label = last_disease_label
                if user_id is not None and row.user_id is None:
                    row.user_id = user_id

                diagnosis_history = list(row.diagnosis_history_json or [])
                if diag_entry:
                    last_entry = diagnosis_history[-1] if diagnosis_history else None
                    same_as_last = (
                        isinstance(last_entry, dict)
                        and last_entry.get("image_diagnosis_label") == diag_entry.get("image_diagnosis_label")
                        and last_entry.get("physical_diagnosis_label") == diag_entry.get("physical_diagnosis_label")
                        and last_entry.get("pathology_top1") == diag_entry.get("pathology_top1")
                        and last_entry.get("last_disease_label") == diag_entry.get("last_disease_label")
                    )
                    if not same_as_last:
                        diagnosis_history.append(diag_entry)
                row.diagnosis_history_json = diagnosis_history

            db.commit()
        finally:
            db.close()


session_memory_store = SessionMemoryStore()
