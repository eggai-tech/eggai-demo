import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Avoid 'RuntimeError: Event loop is closed' across async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
