from __future__ import annotations

import os
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from database import get_db
from user_models import User

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
# 8001 不负责登录，这里仅用于 Bearer Token 解析；前端仍向 8000 登录换 token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或过期的认证令牌",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


def require_roles(*allowed_roles: str) -> Callable:
    allowed = {r.lower() for r in allowed_roles}

    def _dependency(current_user: User = Depends(get_current_user)) -> User:
        role = (current_user.role or "").lower()
        if role not in allowed:
            raise HTTPException(status_code=403, detail="当前账号无权访问该接口")
        return current_user

    return _dependency


require_chat_user = require_roles("user", "doctor", "admin")
require_admin = require_roles("admin")
