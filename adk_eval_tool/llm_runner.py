"""Utilities for running ADK agents with timeout and retry support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3


async def run_agent_with_timeout(
    runner,
    *,
    user_id: str,
    session_id: str,
    new_message,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    session_service=None,
    app_name: str = "",
) -> list:
    """Run an ADK agent with timeout and retry.

    Args:
        runner: ADK Runner instance.
        user_id: User ID for the session.
        session_id: Session ID.
        new_message: Content to send.
        timeout: Timeout in seconds per attempt.
        max_retries: Maximum number of retry attempts.
        session_service: Optional session service for creating new sessions on retry.
        app_name: App name for creating new sessions on retry.

    Returns:
        List of events collected from the agent run.

    Raises:
        TimeoutError: If all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            events = []
            async with asyncio.timeout(timeout):
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=new_message,
                ):
                    events.append(event)
            return events
        except TimeoutError:
            last_error = TimeoutError(
                f"Agent run timed out after {timeout}s (attempt {attempt}/{max_retries})"
            )
            logger.warning("Attempt %d/%d timed out after %ds", attempt, max_retries, timeout)
            # Create a fresh session for retry
            if session_service and app_name and attempt < max_retries:
                try:
                    new_session = await session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )
                    session_id = new_session.id
                except Exception:
                    pass
        except Exception as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d failed: %s", attempt, max_retries, e
            )
            if attempt >= max_retries:
                break
            # Create a fresh session for retry
            if session_service and app_name:
                try:
                    new_session = await session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )
                    session_id = new_session.id
                except Exception:
                    pass

    raise last_error or TimeoutError("Agent run failed after all retries")


async def run_inference_with_timeout(
    eval_service,
    inference_request,
    *,
    timeout: int = 300,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list:
    """Run ADK inference with timeout and retry.

    Args:
        eval_service: LocalEvalService instance.
        inference_request: InferenceRequest to execute.
        timeout: Timeout in seconds per attempt.
        max_retries: Maximum number of retry attempts.

    Returns:
        List of InferenceResult objects.

    Raises:
        TimeoutError: If all retries are exhausted.
    """
    from contextlib import aclosing as Aclosing

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            results = []
            async with asyncio.timeout(timeout):
                async with Aclosing(
                    eval_service.perform_inference(inference_request=inference_request)
                ) as agen:
                    async for result in agen:
                        results.append(result)
            return results
        except TimeoutError:
            last_error = TimeoutError(
                f"Inference timed out after {timeout}s (attempt {attempt}/{max_retries})"
            )
            logger.warning("Inference attempt %d/%d timed out after %ds", attempt, max_retries, timeout)
        except Exception as e:
            last_error = e
            logger.warning("Inference attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt >= max_retries:
                break

    raise last_error or TimeoutError("Inference failed after all retries")
