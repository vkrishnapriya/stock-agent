import pytest
import redis.asyncio as aioredis

REDIS_URL = "redis://localhost:6379/0"


@pytest.fixture
async def redis():
    client = aioredis.from_url(REDIS_URL)
    yield client
    await client.aclose()


async def test_redis_connection(redis):
    pong = await redis.ping()
    assert pong is True


async def test_redis_set_get(redis):
    await redis.set("test_key", "test_value", ex=10)
    value = await redis.get("test_key")
    assert value == b"test_value"


async def test_redis_delete(redis):
    await redis.set("del_key", "del_value", ex=10)
    await redis.delete("del_key")
    value = await redis.get("del_key")
    assert value is None
