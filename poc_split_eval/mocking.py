"""Tool and sub-agent mocking for ADK evaluation inference.

ADK's eval framework does NOT automatically inject tool responses from
eval cases during inference — ``tool_responses`` in ``IntermediateData``
are only used for comparison during the evaluation stage.

This module provides utilities to:

1. **Mock tools** via ``before_tool_callback``: intercept tool calls and
   return canned responses from the eval case, so inference runs without
   real tool backends.

2. **Mock sub-agents**: replace live sub-agents with lightweight stubs
   that return fixed responses, so only the agent-under-test makes real
   LLM calls.

Both mechanisms are applied *before* inference and cleaned up afterwards.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent


# ---------------------------------------------------------------------------
# Tool response map
# ---------------------------------------------------------------------------


ToolResponseMap = dict[str, Any]
"""Mapping of tool_name → canned response dict.

Example::

    {"get_weather": {"city": "London", "temp_c": 15, "condition": "Cloudy"}}
"""


def build_tool_response_map(eval_set_dict: dict) -> ToolResponseMap:
    """Extract a tool_name → response map from an eval set's conversations.

    Scans all invocations in all eval cases for ``tool_uses`` and
    ``tool_responses``.  When a tool_response carries the same ``name``
    as a tool_use, the response dict is captured.  When no explicit
    tool_response exists for a tool_use, a generic success payload is
    generated from the tool_use args.

    Args:
        eval_set_dict: Eval set in snake_case ADK dict format.

    Returns:
        ToolResponseMap keyed by tool name.
    """
    response_map: ToolResponseMap = {}

    for case in eval_set_dict.get("eval_cases", []):
        for inv in case.get("conversation", []):
            idata = inv.get("intermediate_data")
            if not idata:
                continue

            # Index explicit responses by name
            responses_by_name: dict[str, Any] = {}
            for tr in idata.get("tool_responses", []):
                if isinstance(tr, dict) and "name" in tr:
                    responses_by_name[tr["name"]] = tr.get("response", {})

            # Map each tool_use to its response (explicit or synthetic)
            for tu in idata.get("tool_uses", []):
                if not isinstance(tu, dict) or "name" not in tu:
                    continue
                name = tu["name"]
                if name in responses_by_name:
                    response_map[name] = responses_by_name[name]
                elif name not in response_map:
                    # Synthetic success response echoing the args
                    response_map[name] = {
                        "status": "ok",
                        "tool": name,
                        "args_received": tu.get("args", {}),
                    }

    return response_map


# ---------------------------------------------------------------------------
# before_tool_callback factory
# ---------------------------------------------------------------------------


def make_mock_tool_callback(
    response_map: ToolResponseMap,
    *,
    fallback: Any | None = None,
    strict: bool = False,
) -> Callable:
    """Create a ``before_tool_callback`` that returns canned tool responses.

    Args:
        response_map: tool_name → response dict.
        fallback: Response returned for tools NOT in the map.
            ``None`` means "let the real tool execute".
        strict: If True, raise ``KeyError`` for unmapped tools instead of
            falling back.

    Returns:
        An async callback compatible with ``LlmAgent.before_tool_callback``.
    """

    async def _mock_before_tool(tool, args, tool_context):  # noqa: ARG001 — tool_context unused
        name = tool.name if hasattr(tool, "name") else str(tool)
        if name in response_map:
            return response_map[name]
        if strict:
            raise KeyError(
                f"Tool '{name}' not found in response map and strict=True. "
                f"Available: {list(response_map.keys())}"
            )
        return fallback

    return _mock_before_tool


# ---------------------------------------------------------------------------
# Sub-agent stubbing
# ---------------------------------------------------------------------------


def make_stub_agent(
    name: str,
    response_text: str = "Stub response.",
    *,
    model: str = "gemini-2.0-flash",
    description: str = "",
) -> LlmAgent:
    """Create a minimal stub agent that always returns *response_text*.

    The stub has no tools, no sub-agents, and a trivial instruction that
    tells the LLM to reply with the fixed text.  This is useful for
    isolating the agent-under-test from sub-agent LLM calls.

    Args:
        name: Agent name (must match the original sub-agent's name so
            delegation routing works).
        response_text: Fixed text the stub should return.
        model: Model name (a cheap/fast model is fine for stubs).
        description: Agent description (preserved for routing).

    Returns:
        A minimal LlmAgent.
    """
    from google.adk.agents import Agent

    return Agent(
        name=name,
        model=model,
        description=description or f"Stub for {name}",
        instruction=(
            f"You are a test stub. Always reply with exactly: {response_text}"
        ),
    )


# ---------------------------------------------------------------------------
# High-level helpers: install / uninstall mocking on an agent tree
# ---------------------------------------------------------------------------


class MockContext:
    """Captures original state so mocking can be reversed.

    Usage::

        ctx = install_tool_mocks(agent, response_map)
        # ... run inference ...
        ctx.uninstall()
    """

    def __init__(self):
        self._originals: list[tuple[LlmAgent, str, Any]] = []

    def _save(self, agent: LlmAgent, attr: str):
        self._originals.append((agent, attr, getattr(agent, attr)))

    def uninstall(self):
        """Restore all agents to their original state."""
        for agent, attr, original_value in reversed(self._originals):
            setattr(agent, attr, original_value)
        self._originals.clear()


def install_tool_mocks(
    agent: BaseAgent,
    response_map: ToolResponseMap,
    *,
    fallback: Any | None = None,
    recursive: bool = True,
) -> MockContext:
    """Install ``before_tool_callback`` on *agent* (and optionally sub-agents).

    Any existing ``before_tool_callback`` is saved and will be restored
    when ``MockContext.uninstall()`` is called.

    Args:
        agent: Root agent to mock.
        response_map: tool_name → canned response.
        fallback: Response for unmapped tools (None = let real tool run).
        recursive: If True, install on all sub-agents too.

    Returns:
        MockContext for cleanup.
    """
    ctx = MockContext()
    cb = make_mock_tool_callback(response_map, fallback=fallback)

    def _install(a: BaseAgent):
        if isinstance(a, LlmAgent):
            ctx._save(a, "before_tool_callback")
            a.before_tool_callback = cb
        if recursive:
            for sub in a.sub_agents:
                _install(sub)

    _install(agent)
    return ctx


def install_sub_agent_stubs(
    agent: BaseAgent,
    stubs: dict[str, str],
) -> MockContext:
    """Replace named sub-agents with stubs that return fixed text.

    Args:
        agent: Parent agent whose sub-agents should be stubbed.
        stubs: Mapping of sub_agent_name → fixed response text.

    Returns:
        MockContext for cleanup.
    """
    ctx = MockContext()

    def _stub_recursive(a: BaseAgent):
        if not a.sub_agents:
            return
        new_subs = []
        for sub in a.sub_agents:
            if sub.name in stubs:
                ctx._save(a, "sub_agents")
                stub = make_stub_agent(
                    name=sub.name,
                    response_text=stubs[sub.name],
                    description=sub.description or "",
                )
                new_subs.append(stub)
            else:
                _stub_recursive(sub)
                new_subs.append(sub)
        a.sub_agents = new_subs

    _stub_recursive(agent)
    return ctx
