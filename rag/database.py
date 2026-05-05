import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

MYSQL_URL = os.getenv(
    "MYSQL_URL",
    "mysql+pymysql://root:lc43234698@172.22.96.1/heal_agent?charset=utf8mb4",
)

engine = create_engine(
    MYSQL_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
