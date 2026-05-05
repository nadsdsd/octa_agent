from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    """映射 backend/8000 已存在的 users 表，仅供 8001 识别当前用户。"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    # backend 当前是 Enum(user/doctor)，这里用 String 映射，兼容读取已有表
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSessionMemory(Base):
    """8001 专用：按用户绑定唯一会话快照。"""

    __tablename__ = "rag_user_session_memory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)

    history_json = Column(JSON, nullable=False, default=list)
    context_json = Column(JSON, nullable=False, default=dict)
    diagnosis_history_json = Column(JSON, nullable=False, default=list)
    last_disease_label = Column(String(64), nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    owner = relationship("User", lazy="joined")
