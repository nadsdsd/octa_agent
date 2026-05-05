import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 请替换为你的 MySQL 实际账密和数据库名
MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:lc43234698@172.22.96.1/heal_agent?charset=utf8mb4")
engine = create_engine(MYSQL_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# FastAPI 依赖注入，用于在每次请求时获取数据库 Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()