from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import config
from database import SessionLocal
from user_models import UserSessionMemory


def _default_history() -> List[Dict[str, str]]:
    return [{"role": "system", "content": config.SYSTEM_PROMPT}]


def _safe_history(value: Any) -> List[Dict[str, str]]:
    if isinstance(value, list) and value:
        return value
    return _default_history()


def _safe_context(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _build_diagnosis_snapshot(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从当前 context 提炼一条轻量诊断历史，用于保存在 diagnosis_history_json。
    只存有价值的摘要，不重复把整份 context 塞进去。
    """
    last_disease_label = context.get("last_disease_label")
    image_label = context.get("image_diagnosis_label")
    physical_label = context.get("physical_diagnosis_label")
    pathology = context.get("pathology_diagnosis_result") or {}

    pathology_top1 = None
    if isinstance(pathology, dict):
        top3 = pathology.get("top_3_diagnoses") or []
        if isinstance(top3, list) and top3:
            pathology_top1 = top3[0]

    has_any_result = any([
        last_disease_label,
        image_label,
        physical_label,
        pathology_top1,
        context.get("diagnosis_result"),
        context.get("physical_metrics"),
    ])

    if not has_any_result:
        return None

    return {
        "ts": datetime.utcnow().isoformat(),
        "image_diagnosis_label": image_label,
        "physical_diagnosis_label": physical_label,
        "pathology_top1": pathology_top1,
        "last_disease_label": last_disease_label,
        "blocked": bool(context.get("blocked", False)),
    }


class SessionMemoryStore:
    """
    规则：
    1. Redis 是热缓存
    2. MySQL(user_session_memory) 是持久化恢复源
    3. 所有会话严格按 user_id 归属
    4. 兼容旧脏数据：如果表里已有 session_id 对应行但 user_id 为空，会自动回填 user_id
    """

    def __init__(self):
        pass

    @staticmethod
    def build_session_id(user_id: int) -> str:
        # 保持和你当前 8001 版本一致的“一用户一会话”策略
        return f"user_{user_id}_default"

    @staticmethod
    def _redis_key(session_id: str) -> str:
        return f"agent_session:{session_id}"

    @staticmethod
    def _get_redis_client():
        return config.get_redis_client()

    def load_user_state(self, user_id: int) -> Dict[str, Any]:
        session_id = self.build_session_id(user_id)
        redis_key = self._redis_key(session_id)

        # 1. Redis 优先
        try:
            raw = self._get_redis_client().get(redis_key)
            if raw:
                data = json.loads(raw)
                return {
                    "history": _safe_history(data.get("history")),
                    "context": _safe_context(data.get("context")),
                }
        except Exception:
            # Redis 出问题直接回源 MySQL，不中断主链路
            pass

        # 2. MySQL 回源
        db = SessionLocal()
        try:
            # 先按 user_id 查
            row = (
                db.query(UserSessionMemory)
                .filter(UserSessionMemory.user_id == user_id)
                .first()
            )

            # 如果没有，再按 session_id 查，兼容历史旧数据（user_id 为空的行）
            if row is None:
                row = (
                    db.query(UserSessionMemory)
                    .filter(UserSessionMemory.session_id == session_id)
                    .first()
                )
                if row is not None and row.user_id != user_id:
                    row.user_id = user_id
                    db.commit()
                    db.refresh(row)

            if row is None:
                state = {
                    "history": _default_history(),
                    "context": {},
                }
            else:
                state = {
                    "history": _safe_history(row.history_json),
                    "context": _safe_context(row.context_json),
                }

            # 3. 回填 Redis
            try:
                self._get_redis_client().setex(
                    redis_key,
                    config.SESSION_TTL_SECONDS,
                    json.dumps(state, ensure_ascii=False),
                )
            except Exception:
                pass

            return state
        finally:
            db.close()

    def save_user_state(self, user_id: int, state: Dict[str, Any]) -> None:
        session_id = self.build_session_id(user_id)
        redis_key = self._redis_key(session_id)

        history = _safe_history(state.get("history"))
        context = _safe_context(state.get("context"))
        last_disease_label = context.get("last_disease_label")

        db = SessionLocal()
        try:
            # 先按 user_id 查，确保真正按用户写
            row = (
                db.query(UserSessionMemory)
                .filter(UserSessionMemory.user_id == user_id)
                .first()
            )

            # 如果没有，再按 session_id 查，兼容旧行并回填 user_id
            if row is None:
                row = (
                    db.query(UserSessionMemory)
                    .filter(UserSessionMemory.session_id == session_id)
                    .first()
                )
                if row is not None and row.user_id != user_id:
                    row.user_id = user_id

            # 处理 diagnosis_history_json
            diagnosis_snapshot = _build_diagnosis_snapshot(context)

            if row is None:
                diagnosis_history: List[Dict[str, Any]] = []
                if diagnosis_snapshot:
                    diagnosis_history.append(diagnosis_snapshot)

                row = UserSessionMemory(
                    user_id=user_id,
                    session_id=session_id,
                    history_json=history,
                    context_json=context,
                    diagnosis_history_json=diagnosis_history,
                    last_disease_label=last_disease_label,
                )
                db.add(row)
            else:
                diagnosis_history = row.diagnosis_history_json or []
                if not isinstance(diagnosis_history, list):
                    diagnosis_history = []

                # 只有有新诊断摘要时才追加，避免无意义重复
                if diagnosis_snapshot:
                    should_append = True
                    if diagnosis_history:
                        last_item = diagnosis_history[-1]
                        if isinstance(last_item, dict):
                            should_append = (
                                last_item.get("last_disease_label") != diagnosis_snapshot.get("last_disease_label")
                                or last_item.get("image_diagnosis_label") != diagnosis_snapshot.get("image_diagnosis_label")
                                or last_item.get("physical_diagnosis_label") != diagnosis_snapshot.get("physical_diagnosis_label")
                                or last_item.get("pathology_top1") != diagnosis_snapshot.get("pathology_top1")
                            )
                    if should_append:
                        diagnosis_history.append(diagnosis_snapshot)

                row.user_id = user_id
                row.session_id = session_id
                row.history_json = history
                row.context_json = context
                row.diagnosis_history_json = diagnosis_history
                row.last_disease_label = last_disease_label

            db.commit()

            # 同步刷新 Redis
            try:
                self._get_redis_client().setex(
                    redis_key,
                    config.SESSION_TTL_SECONDS,
                    json.dumps({"history": history, "context": context}, ensure_ascii=False),
                )
            except Exception:
                pass
        finally:
            db.close()

    def clear_user_state(self, user_id: int) -> None:
        session_id = self.build_session_id(user_id)
        redis_key = self._redis_key(session_id)

        db = SessionLocal()
        try:
            row = (
                db.query(UserSessionMemory)
                .filter(UserSessionMemory.user_id == user_id)
                .first()
            )
            if row is not None:
                row.history_json = _default_history()
                row.context_json = {}
                db.commit()
        finally:
            db.close()

        try:
            self._get_redis_client().delete(redis_key)
        except Exception:
            pass


session_memory_store = SessionMemoryStore()