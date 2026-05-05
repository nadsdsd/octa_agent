import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Enum, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from config.database import Base

class RoleEnum(str, enum.Enum):
    USER = "user"       # 普通用户
    DOCTOR = "doctor"   # 医生

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.USER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 建立与会话表的一对多关系
    sessions = relationship("UserSession", back_populates="owner", cascade="all, delete-orphan")


class UserSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False) # 对应 Redis 中的 session_id
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="sessions")