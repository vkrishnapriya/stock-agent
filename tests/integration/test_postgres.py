import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/stockagent"


@pytest.fixture
async def engine():
    e = create_async_engine(DATABASE_URL)
    yield e
    await e.dispose()


async def test_postgres_connection(engine):
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


async def test_postgres_version(engine):
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT version()"))
        version = result.scalar()
        assert "PostgreSQL" in version
        print(f"\n{version}")
