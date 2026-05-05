import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from config.database import get_db
# 把这行：
# from models.user_models import User, UserSession
# 改为：
from db_models.user_models import User, UserSession
from schemas.user_schemas import UserCreate, UserLogin, SessionResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# 密码加密工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

@router.post("/register")
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    hashed_pwd = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, password_hash=hashed_pwd, role=user_data.role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "注册成功", "user_id": new_user.id}

@router.post("/login", response_model=SessionResponse)
def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    # 1. 验证用户
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if not db_user or not verify_password(user_data.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    # 2. 每次登录创建全新的 session_id
    new_session_id = f"sess_{uuid.uuid4().hex}"
    
    # 3. 将新会话绑定到该用户，并存入 MySQL 永久记录
    new_session = UserSession(session_id=new_session_id, user_id=db_user.id)
    db.add(new_session)
    db.commit()
    
    # 提示：你不需要在这里操作 Redis。
    # Redis 的缓存会在用户第一次调用 /api/chat 传入新 session_id 时自动创建。
    
    return SessionResponse(
        session_id=new_session_id,
        message="登录成功，已分配全新会话",
        role=db_user.role.value
    )