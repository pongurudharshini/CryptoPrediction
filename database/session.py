"""
Database Session Manager.
Configures the SQLAlchemy engine and provides a session maker for database interactions.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import settings

# Create database engine. 
# connect_args={"check_same_thread": False} is required for SQLite in multithreaded/async FastAPI apps.
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency generator for FastAPI to safely yield database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()