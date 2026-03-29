"""Tests for LLM runner timeout and retry utilities."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from adk_eval_tool.llm_runner import run_agent_with_timeout, run_inference_with_timeout


class FakeEvent:
    def __init__(self, text=None):
        self.content = MagicMock()
        if text:
            part = MagicMock()
            part.text = text
            part.function_call = None
            self.content.parts = [part]
        else:
            self.content.parts = []


class FakeRunner:
    def __init__(self, events=None, delay=0, fail_times=0):
        self._events = events or [FakeEvent("hello")]
        self._delay = delay
        self._fail_times = fail_times
        self._call_count = 0

    async def run_async(self, **kwargs):
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise RuntimeError(f"Simulated failure {self._call_count}")
        if self._delay:
            await asyncio.sleep(self._delay)
        for event in self._events:
            yield event


class FakeSessionService:
    async def create_session(self, **kwargs):
        session = MagicMock()
        session.id = f"sess-{id(session)}"
        return session


@pytest.mark.asyncio
async def test_run_agent_success():
    runner = FakeRunner(events=[FakeEvent("result")])
    events = await run_agent_with_timeout(
        runner, user_id="u", session_id="s", new_message="msg", timeout=5
    )
    assert len(events) == 1


@pytest.mark.asyncio
async def test_run_agent_timeout():
    runner = FakeRunner(delay=10)
    with pytest.raises(TimeoutError, match="timed out"):
        await run_agent_with_timeout(
            runner, user_id="u", session_id="s", new_message="msg",
            timeout=1, max_retries=1,
        )


@pytest.mark.asyncio
async def test_run_agent_retry_on_failure():
    runner = FakeRunner(events=[FakeEvent("ok")], fail_times=2)
    events = await run_agent_with_timeout(
        runner, user_id="u", session_id="s", new_message="msg",
        timeout=5, max_retries=3,
        session_service=FakeSessionService(), app_name="test",
    )
    assert len(events) == 1
    assert runner._call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_run_agent_exhausts_retries():
    runner = FakeRunner(fail_times=5)
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await run_agent_with_timeout(
            runner, user_id="u", session_id="s", new_message="msg",
            timeout=5, max_retries=3,
            session_service=FakeSessionService(), app_name="test",
        )
    assert runner._call_count == 3


@pytest.mark.asyncio
async def test_run_inference_timeout():
    class HangingService:
        async def perform_inference(self, **kwargs):
            await asyncio.sleep(10)
            yield  # never reached

    with pytest.raises(TimeoutError, match="timed out"):
        await run_inference_with_timeout(
            HangingService(), MagicMock(),
            timeout=1, max_retries=1,
        )
