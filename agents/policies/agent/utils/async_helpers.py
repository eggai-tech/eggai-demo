import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

from libraries.observability.logger import get_console_logger

logger = get_console_logger("async_helpers")

T = TypeVar('T')


def run_async_safe(coro: Coroutine[Any, Any, T]) -> T:
    """Run async code safely in both sync and async contexts.

    Detects whether we're in an async or sync context and handles
    coroutine execution appropriately. In async context, uses a
    thread pool since we can't nest event loops.
    """
    try:
        asyncio.get_running_loop()
        logger.debug("Running coroutine in thread pool from async context")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    except RuntimeError:
        logger.debug("Running coroutine directly in sync context")
        return asyncio.run(coro)


async def to_async_iterator(sync_iterable):
    for item in sync_iterable:
        yield item


def ensure_coroutine(func_or_coro):
    if asyncio.iscoroutine(func_or_coro):
        return func_or_coro

    async def _wrapper():
        return func_or_coro

    return _wrapper()
