from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String
from config.database import Base


class SessionMemory(Base):
    """LangGraph(RAG 8001) 的持久化会话快照表。

    说明：
    - 不修改 backend 现有 users / sessions 逻辑；
    - 仅新增一张快照表，用于 Redis 丢失时回源恢复；
    - session_id 直接与 8001 的 session_id 对齐；
    - user_id 先不强依赖，避免当前双服务架构下额外改前端登录态传递。
    """

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
