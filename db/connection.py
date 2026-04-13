# db/connection.py
"""
Lumora Intelligence — PostgreSQL bağlantısı (SQLAlchemy).
"""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

import config

logger = logging.getLogger(__name__)

# ─── Engine ──────────────────────────────────────────────────────────────────
engine = create_engine(
    config.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    """Context dışında kullanım için session döndürür."""
    return SessionLocal()


def check_connection() -> bool:
    """DB bağlantısını test eder. True → başarılı."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB bağlantı hatası: {e}")
        return False
