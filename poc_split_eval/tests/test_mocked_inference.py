"""Tests for inference with mocked tools and stubbed sub-agents.

These tests build realistic InferenceResult artifacts with mocked
tool responses, then evaluate them — verifying that the full
mock → persist → evaluate roundtrip works.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from google.adk.evaluation.base_eval_service import (
    InferenceResult,
    InferenceStatus,
)
from google.adk.evaluation.eval_case import Invocation, IntermediateData
from google.genai import types as genai_types

from poc_split_eval.mocking import build_tool_response_map
from poc_split_eval.schemas import InferenceArtifact, InferenceBundle
from poc_split_eval.inference import save_inference_bundle, load_inference_bundle
from poc_split_eval.evaluation import run_evaluation_from_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_content(role: str, text: str) -> genai_types.Content:
    return genai_types.Content(role=role, parts=[genai_types.Part(text=text)])


def _make_invocation(
    inv_id: str,
    user_text: str,
    response_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Invocation:
    intermediate = None
    if tool_calls:
        func_calls = [
            genai_types.FunctionCall(name=tc["name"], args=tc.get("args", {}))
            for tc in tool_calls
        ]
        intermediate = IntermediateData(
            tool_uses=func_calls,
            tool_responses=[],
            intermediate_responses=[],
        )
    return Invocation(
        invocation_id=inv_id,
        user_content=_make_content("user", user_text),
        final_response=_make_content("model", response_text),
        intermediate_data=intermediate,
    )


# ---------------------------------------------------------------------------
# Eval set fixtures with tool responses (for mocking)
# ---------------------------------------------------------------------------


def _weather_eval_set_with_responses() -> dict:
    """Eval set where tool_responses provide the expected mock data."""
    return {
        "eval_set_id": "weather__mocked",
        "eval_cases": [
            {
                "eval_id": "london_weather_mocked",
                "conversation": [
                    {
                        "invocation_id": "inv-1",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "What's the weather in London?"}],
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
                                        "humidity": 78,
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
        ],
    }


def _multi_agent_eval_set() -> dict:
    """Eval set for multi-agent with delegated tool calls."""
    return {
        "eval_set_id": "travel__mocked",
        "eval_cases": [
            {
                "eval_id": "flight_search_mocked",
                "conversation": [
                    {
                        "invocation_id": "inv-1",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "Find flights from London to Paris"}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {
                                    "name": "search_flights",
                                    "args": {
                                        "origin": "London",
                                        "destination": "Paris",
                                    },
                                },
                            ],
                            "tool_responses": [
                                {
                                    "name": "search_flights",
                                    "response": {
                                        "flights": [
                                            {
                                                "id": "FL-101",
                                                "airline": "MockAir",
                                                "price": 200,
                                            }
                                        ]
                                    },
                                },
                            ],
                            "intermediate_responses": [],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [
                                {"text": "Found flight FL-101 on MockAir for $200."}
                            ],
                        },
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Build artifacts with mocked tool calls
# ---------------------------------------------------------------------------


def _build_mocked_weather_artifact() -> InferenceArtifact:
    """Artifact where agent used get_weather with mocked response."""
    eval_set = _weather_eval_set_with_responses()

    actual_inv = _make_invocation(
        inv_id="inv-1",
        user_text="What's the weather in London?",
        response_text="London is 15°C and cloudy.",
        tool_calls=[{"name": "get_weather", "args": {"city": "London"}}],
    )
    ir = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather__mocked",
        eval_case_id="london_weather_mocked",
        session_id="sess-mock-001",
        status=InferenceStatus.SUCCESS,
        inferences=[actual_inv],
    )
    return InferenceArtifact(
        eval_set_json=eval_set,
        inference_result_json=json.loads(ir.model_dump_json()),
        session_id="sess-mock-001",
    )


def _build_mocked_flight_artifact() -> InferenceArtifact:
    """Artifact where agent delegated to flight sub-agent (stubbed)."""
    eval_set = _multi_agent_eval_set()

    actual_inv = _make_invocation(
        inv_id="inv-1",
        user_text="Find flights from London to Paris",
        response_text="Found flight FL-101 on MockAir for $200.",
        tool_calls=[
            {
                "name": "search_flights",
                "args": {"origin": "London", "destination": "Paris"},
            }
        ],
    )
    ir = InferenceResult(
        app_name="eval_app",
        eval_set_id="travel__mocked",
        eval_case_id="flight_search_mocked",
        session_id="sess-mock-002",
        status=InferenceStatus.SUCCESS,
        inferences=[actual_inv],
    )
    return InferenceArtifact(
        eval_set_json=eval_set,
        inference_result_json=json.loads(ir.model_dump_json()),
        session_id="sess-mock-002",
    )


def _build_wrong_tool_mocked_artifact() -> InferenceArtifact:
    """Artifact where agent called wrong tool despite mocking."""
    eval_set = _weather_eval_set_with_responses()

    # Agent called get_forecast instead of expected get_weather
    actual_inv = _make_invocation(
        inv_id="inv-1",
        user_text="What's the weather in London?",
        response_text="Here's the forecast for London.",
        tool_calls=[{"name": "get_forecast", "args": {"city": "London", "days": 3}}],
    )
    ir = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather__mocked",
        eval_case_id="london_weather_mocked",
        session_id="sess-mock-003",
        status=InferenceStatus.SUCCESS,
        inferences=[actual_inv],
    )
    return InferenceArtifact(
        eval_set_json=eval_set,
        inference_result_json=json.loads(ir.model_dump_json()),
        session_id="sess-mock-003",
    )


# ---------------------------------------------------------------------------
# Tests: tool response map extraction from eval sets
# ---------------------------------------------------------------------------


class TestToolResponseMapFromEvalSet:

    def test_weather_eval_set_has_explicit_responses(self):
        m = build_tool_response_map(_weather_eval_set_with_responses())
        assert "get_weather" in m
        assert m["get_weather"]["city"] == "London"
        assert m["get_weather"]["temp_c"] == 15
        assert m["get_weather"]["humidity"] == 78

    def test_multi_agent_eval_set_has_flight_responses(self):
        m = build_tool_response_map(_multi_agent_eval_set())
        assert "search_flights" in m
        flights = m["search_flights"]["flights"]
        assert len(flights) == 1
        assert flights[0]["airline"] == "MockAir"


# ---------------------------------------------------------------------------
# Tests: evaluation of mocked inference artifacts
# ---------------------------------------------------------------------------


class TestMockedInferenceEvaluation:

    @pytest.mark.asyncio
    async def test_mocked_tool_correct_call_passes(self):
        """Agent called the right mocked tool → PASSED."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_mocked_weather_artifact()],
            metadata={"mock_tools": True},
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        assert results[0].status == "PASSED"
        assert results[0].overall_scores["tool_trajectory_avg_score"] == 1.0

    @pytest.mark.asyncio
    async def test_mocked_tool_wrong_call_fails(self):
        """Agent called wrong tool despite mocking → FAILED."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_wrong_tool_mocked_artifact()],
            metadata={"mock_tools": True},
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        assert results[0].status == "FAILED"

    @pytest.mark.asyncio
    async def test_mocked_multi_agent_passes(self):
        """Agent delegated to stubbed sub-agent with correct tool → PASSED."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_mocked_flight_artifact()],
            metadata={"stub_sub_agents": ["flight_agent"]},
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        assert results[0].status == "PASSED"
        assert results[0].overall_scores["tool_trajectory_avg_score"] == 1.0


# ---------------------------------------------------------------------------
# Tests: full roundtrip with mocked artifacts
# ---------------------------------------------------------------------------


class TestMockedRoundtrip:

    @pytest.mark.asyncio
    async def test_save_load_evaluate_mocked(self, tmp_path):
        """Build mocked artifact → save → load → evaluate."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_mocked_weather_artifact()],
            metadata={"mock_tools": True},
        )
        path = str(tmp_path / "mocked_bundle.json")
        save_inference_bundle(bundle, path)

        loaded = load_inference_bundle(path)
        results = await run_evaluation_from_bundle(
            loaded,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        assert results[0].status == "PASSED"

    @pytest.mark.asyncio
    async def test_mixed_mocked_and_real(self):
        """Bundle with both passing and failing artifacts evaluated together."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[
                _build_mocked_weather_artifact(),  # correct tool → PASS
                _build_wrong_tool_mocked_artifact(),  # wrong tool → FAIL
                _build_mocked_flight_artifact(),  # correct sub-agent tool → PASS
            ],
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 3
        passed = [r for r in results if r.status == "PASSED"]
        failed = [r for r in results if r.status == "FAILED"]
        assert len(passed) == 2
        assert len(failed) == 1
        assert failed[0].eval_case_id == "london_weather_mocked"

    @pytest.mark.asyncio
    async def test_per_invocation_shows_mocked_tools(self):
        """Per-invocation data should show both expected and actual tools."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_mocked_weather_artifact()],
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        inv = results[0].per_invocation[0]
        assert inv["actual_tool_calls"][0]["name"] == "get_weather"
        assert inv["expected_tool_calls"][0]["name"] == "get_weather"
        assert inv["actual_tool_calls"][0]["args"]["city"] == "London"

    @pytest.mark.asyncio
    async def test_re_evaluate_mocked_with_different_threshold(self):
        """Same mocked inference, different threshold → different result."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_wrong_tool_mocked_artifact()],
        )

        # Strict threshold → FAIL
        r1 = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert r1[0].status == "FAILED"

        # Lenient threshold → PASS
        r2 = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.0},
            ],
        )
        assert r2[0].status == "PASSED"
