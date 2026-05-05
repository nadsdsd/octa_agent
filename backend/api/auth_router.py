import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from passlib.context import CryptContext
from jose import jwt
from sqlalchemy.orm import Session

from config.database import get_db
from db_models.user_models import User
from schemas.user_schemas import UserCreate, UserLogin, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


@router.post("/register", response_model=TokenResponse)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    username = (user_data.username or "").strip()
    password = (user_data.password or "").strip()
    role = (user_data.role or "doctor").strip().lower()

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="用户名已存在")

    new_user = User(
        username=username,
        password_hash=get_password_hash(password),
        role=role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(username=new_user.username, role=new_user.role.value if hasattr(new_user.role, "value") else str(new_user.role))

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=new_user.username,
        role=new_user.role.value if hasattr(new_user.role, "value") else str(new_user.role),
    )


@router.post("/login", response_model=TokenResponse)
def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    username = (user_data.username or "").strip()
    password = (user_data.password or "").strip()

    db_user = db.query(User).filter(User.username == username).first()
    if not db_user or not verify_password(password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    role = db_user.role.value if hasattr(db_user.role, "value") else str(db_user.role)
    token = create_access_token(username=db_user.username, role=role)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=db_user.username,
        role=role,
    )
