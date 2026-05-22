"""MacPro AI — Async database engine and session factory."""
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        connect_args = {}
        if "sqlite" in settings.database_url:
            connect_args = {"check_same_thread": False}
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            connect_args=connect_args,
        )
    return _engine


async def init_db() -> None:
    """Create all tables if they don't exist."""
    # Import models so SQLModel sees them
    from src.models.schema import Document, Page, Asset  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database initialized.")


def get_session_factory():
    engine = get_engine()
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
