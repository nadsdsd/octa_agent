from pydantic import BaseModel
from typing import Optional
from datetime import datetime
# 把这行：
# from models.user_models import RoleEnum
# 改为：
from db_models.user_models import RoleEnum

class UserCreate(BaseModel):
    username: str
    password: str
    role: RoleEnum = RoleEnum.USER

class UserLogin(BaseModel):
    username: str
    password: str

class SessionResponse(BaseModel):
    session_id: str
    message: str
    role: str

class UserResponse(BaseModel):
    id: int
    username: str
    role: RoleEnum
    created_at: datetime

    # 将原来的 class Config: orm_mode = True 替换为 Pydantic V2 的推荐写法
    model_config = {"from_attributes": True}