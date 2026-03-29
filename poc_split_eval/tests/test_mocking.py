"""Tests for tool and sub-agent mocking utilities."""

from __future__ import annotations

import json
from typing import Any

import pytest

from google.adk.agents import Agent
from google.adk.agents.llm_agent import LlmAgent

from poc_split_eval.mocking import (
    MockContext,
    ToolResponseMap,
    build_tool_response_map,
    install_sub_agent_stubs,
    install_tool_mocks,
    make_mock_tool_callback,
    make_stub_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent_with_tools() -> Agent:
    """Create a simple agent with dummy tools."""

    def get_weather(city: str) -> dict:
        """Get current weather."""
        return {"city": city, "temp_c": 20}

    def get_forecast(city: str, days: int = 3) -> dict:
        """Get weather forecast."""
        return {"city": city, "days": days, "forecast": []}

    return Agent(
        name="weather_agent",
        model="gemini-2.0-flash",
        instruction="You help with weather.",
        tools=[get_weather, get_forecast],
    )


def _make_multi_agent() -> Agent:
    """Create a multi-agent with sub-agents that have tools."""

    def search_flights(origin: str, destination: str) -> dict:
        """Search flights."""
        return {"flights": []}

    def search_hotels(city: str) -> dict:
        """Search hotels."""
        return {"hotels": []}

    flight_agent = Agent(
        name="flight_agent",
        model="gemini-2.0-flash",
        description="Flight specialist",
        instruction="Search flights.",
        tools=[search_flights],
    )
    hotel_agent = Agent(
        name="hotel_agent",
        model="gemini-2.0-flash",
        description="Hotel specialist",
        instruction="Search hotels.",
        tools=[search_hotels],
    )
    return Agent(
        name="coordinator",
        model="gemini-2.0-flash",
        instruction="Coordinate travel.",
        sub_agents=[flight_agent, hotel_agent],
    )


def _sample_eval_set_dict() -> dict:
    """An eval set with tool_uses and tool_responses."""
    return {
        "eval_set_id": "weather_eval",
        "eval_cases": [
            {
                "eval_id": "case_1",
                "conversation": [
                    {
                        "invocation_id": "inv-1",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "Weather in London?"}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {"name": "get_weather", "args": {"city": "London"}},
                            ],
                            "tool_responses": [
                                {
                                    "name": "get_weather",
                                    "response": {
                                        "city": "London",
                                        "temp_c": 15,
                                        "condition": "Cloudy",
                                    },
                                },
                            ],
                            "intermediate_responses": [],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [{"text": "London is 15°C and cloudy."}],
                        },
                    },
                ],
            },
            {
                "eval_id": "case_2",
                "conversation": [
                    {
                        "invocation_id": "inv-2",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "5-day forecast for Tokyo?"}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {
                                    "name": "get_forecast",
                                    "args": {"city": "Tokyo", "days": 5},
                                },
                            ],
                            # No explicit tool_responses → synthetic fallback
                            "tool_responses": [],
                            "intermediate_responses": [],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [{"text": "Here's the 5-day forecast."}],
                        },
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: build_tool_response_map
# ---------------------------------------------------------------------------


class TestBuildToolResponseMap:

    def test_extracts_explicit_responses(self):
        m = build_tool_response_map(_sample_eval_set_dict())
        assert "get_weather" in m
        assert m["get_weather"]["city"] == "London"
        assert m["get_weather"]["temp_c"] == 15

    def test_generates_synthetic_for_missing_responses(self):
        m = build_tool_response_map(_sample_eval_set_dict())
        assert "get_forecast" in m
        # Synthetic response should include status and args
        assert m["get_forecast"]["status"] == "ok"
        assert m["get_forecast"]["args_received"]["city"] == "Tokyo"

    def test_empty_eval_set(self):
        m = build_tool_response_map({"eval_set_id": "empty", "eval_cases": []})
        assert m == {}

    def test_no_intermediate_data(self):
        es = {
            "eval_set_id": "no_tools",
            "eval_cases": [
                {
                    "eval_id": "c1",
                    "conversation": [
                        {
                            "invocation_id": "inv-1",
                            "user_content": {"role": "user", "parts": [{"text": "hi"}]},
                        }
                    ],
                }
            ],
        }
        m = build_tool_response_map(es)
        assert m == {}


# ---------------------------------------------------------------------------
# Tests: make_mock_tool_callback
# ---------------------------------------------------------------------------


class TestMockToolCallback:

    @pytest.mark.asyncio
    async def test_returns_mapped_response(self):
        cb = make_mock_tool_callback({"my_tool": {"result": 42}})

        class FakeTool:
            name = "my_tool"

        result = await cb(FakeTool(), {}, None)
        assert result == {"result": 42}

    @pytest.mark.asyncio
    async def test_returns_none_for_unmapped(self):
        cb = make_mock_tool_callback({"my_tool": {"result": 42}})

        class FakeTool:
            name = "other_tool"

        result = await cb(FakeTool(), {}, None)
        assert result is None  # let real tool run

    @pytest.mark.asyncio
    async def test_returns_fallback_for_unmapped(self):
        cb = make_mock_tool_callback(
            {"my_tool": {"result": 42}},
            fallback={"error": "not available"},
        )

        class FakeTool:
            name = "other_tool"

        result = await cb(FakeTool(), {}, None)
        assert result == {"error": "not available"}

    @pytest.mark.asyncio
    async def test_strict_raises_for_unmapped(self):
        cb = make_mock_tool_callback({"my_tool": {"result": 42}}, strict=True)

        class FakeTool:
            name = "other_tool"

        with pytest.raises(KeyError, match="other_tool"):
            await cb(FakeTool(), {}, None)


# ---------------------------------------------------------------------------
# Tests: install_tool_mocks / uninstall
# ---------------------------------------------------------------------------


class TestInstallToolMocks:

    def test_installs_callback_on_agent(self):
        agent = _make_agent_with_tools()
        assert agent.before_tool_callback is None

        ctx = install_tool_mocks(agent, {"get_weather": {"output": 10}})
        assert agent.before_tool_callback is not None

        ctx.uninstall()
        assert agent.before_tool_callback is None

    def test_installs_recursively(self):
        agent = _make_multi_agent()
        flight = agent.sub_agents[0]
        hotel = agent.sub_agents[1]

        ctx = install_tool_mocks(
            agent,
            {"search_flights": {"flights": []}, "search_hotels": {"hotels": []}},
            recursive=True,
        )
        assert flight.before_tool_callback is not None
        assert hotel.before_tool_callback is not None

        ctx.uninstall()
        assert flight.before_tool_callback is None
        assert hotel.before_tool_callback is None

    def test_non_recursive_only_affects_root(self):
        agent = _make_multi_agent()
        flight = agent.sub_agents[0]

        ctx = install_tool_mocks(
            agent,
            {"search_flights": {"flights": []}},
            recursive=False,
        )
        # Coordinator has no tools so callback is set but sub-agents untouched
        assert flight.before_tool_callback is None

        ctx.uninstall()

    def test_preserves_original_callback(self):
        agent = _make_agent_with_tools()
        original_cb = lambda tool, args, ctx: None  # noqa: E731
        agent.before_tool_callback = original_cb

        ctx = install_tool_mocks(agent, {"get_weather": {"output": 10}})
        assert agent.before_tool_callback is not original_cb

        ctx.uninstall()
        assert agent.before_tool_callback is original_cb


# ---------------------------------------------------------------------------
# Tests: sub-agent stubbing
# ---------------------------------------------------------------------------


class TestSubAgentStubs:

    def test_make_stub_agent(self):
        stub = make_stub_agent("flight_agent", "No flights available.")
        assert stub.name == "flight_agent"
        assert isinstance(stub, LlmAgent)
        assert "No flights available" in stub.instruction

    def test_install_sub_agent_stubs(self):
        agent = _make_multi_agent()
        original_flight = agent.sub_agents[0]
        assert original_flight.name == "flight_agent"
        assert len(original_flight.tools) == 1

        ctx = install_sub_agent_stubs(
            agent,
            {"flight_agent": "Mock: No flights today."},
        )

        # flight_agent should be replaced
        new_flight = next(s for s in agent.sub_agents if s.name == "flight_agent")
        assert len(new_flight.tools) == 0
        assert "Mock: No flights today" in new_flight.instruction

        # hotel_agent should be unchanged
        hotel = next(s for s in agent.sub_agents if s.name == "hotel_agent")
        assert len(hotel.tools) == 1

        ctx.uninstall()

        # Originals should be restored
        restored_flight = next(s for s in agent.sub_agents if s.name == "flight_agent")
        assert len(restored_flight.tools) == 1

    def test_stub_preserves_description(self):
        agent = _make_multi_agent()
        original_desc = agent.sub_agents[0].description

        ctx = install_sub_agent_stubs(
            agent,
            {"flight_agent": "Stub response"},
        )
        stub = next(s for s in agent.sub_agents if s.name == "flight_agent")
        assert stub.description == original_desc

        ctx.uninstall()

    def test_stub_unknown_name_is_noop(self):
        agent = _make_multi_agent()
        original_names = [s.name for s in agent.sub_agents]

        ctx = install_sub_agent_stubs(agent, {"nonexistent_agent": "stub"})
        assert [s.name for s in agent.sub_agents] == original_names

        ctx.uninstall()


# ---------------------------------------------------------------------------
# Tests: combined mocking (tools + sub-agents)
# ---------------------------------------------------------------------------


class TestCombinedMocking:

    def test_both_tools_and_stubs(self):
        agent = _make_multi_agent()
        hotel = next(s for s in agent.sub_agents if s.name == "hotel_agent")

        tool_ctx = install_tool_mocks(
            agent,
            {"search_hotels": {"hotels": [{"name": "Mock Hotel"}]}},
            recursive=True,
        )
        stub_ctx = install_sub_agent_stubs(
            agent,
            {"flight_agent": "No flights available."},
        )

        # hotel_agent should have mock callback
        new_hotel = next(s for s in agent.sub_agents if s.name == "hotel_agent")
        assert new_hotel.before_tool_callback is not None

        # flight_agent should be stubbed
        new_flight = next(s for s in agent.sub_agents if s.name == "flight_agent")
        assert len(new_flight.tools) == 0

        stub_ctx.uninstall()
        tool_ctx.uninstall()

        # Verify full restoration
        restored_flight = next(s for s in agent.sub_agents if s.name == "flight_agent")
        assert len(restored_flight.tools) == 1
        restored_hotel = next(s for s in agent.sub_agents if s.name == "hotel_agent")
        assert restored_hotel.before_tool_callback is None

    def test_build_map_and_install_from_eval_set(self):
        """End-to-end: extract response map from eval set and install."""
        agent = _make_agent_with_tools()
        response_map = build_tool_response_map(_sample_eval_set_dict())

        ctx = install_tool_mocks(agent, response_map)
        assert agent.before_tool_callback is not None

        ctx.uninstall()
        assert agent.before_tool_callback is None
