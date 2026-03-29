"""Tests for the split inference/evaluation POC.

These tests verify that:
1. InferenceResults can be serialised to JSON and deserialised
2. Evaluation can score pre-built InferenceResults without a live agent
3. The full roundtrip (build artifact → serialise → evaluate) works
4. Re-evaluation with different metrics produces different results
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

from poc_split_eval.schemas import InferenceArtifact, InferenceBundle
from poc_split_eval.evaluation import (
    build_eval_metrics,
    run_evaluation_from_bundle,
    EvalResult,
)
from poc_split_eval.inference import save_inference_bundle, load_inference_bundle


# ---------------------------------------------------------------------------
# Fixtures: build realistic InferenceResult + EvalSet without running agent
# ---------------------------------------------------------------------------


def _make_content(role: str, text: str) -> genai_types.Content:
    return genai_types.Content(
        role=role, parts=[genai_types.Part(text=text)]
    )


def _make_invocation(
    inv_id: str,
    user_text: str,
    response_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Invocation:
    """Build an Invocation with optional tool calls."""
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


def _make_eval_set_dict(
    eval_set_id: str,
    cases: list[dict],
) -> dict:
    """Build an EvalSet dict in snake_case ADK format."""
    return {"eval_set_id": eval_set_id, "eval_cases": cases}


def _make_eval_case_dict(
    eval_id: str,
    conversation: list[dict],
) -> dict:
    return {"eval_id": eval_id, "conversation": conversation}


def _make_conversation_turn(
    inv_id: str,
    user_text: str,
    response_text: str,
    tool_uses: list[dict] | None = None,
) -> dict:
    turn: dict[str, Any] = {
        "invocation_id": inv_id,
        "user_content": {"role": "user", "parts": [{"text": user_text}]},
        "final_response": {"role": "model", "parts": [{"text": response_text}]},
    }
    if tool_uses:
        turn["intermediate_data"] = {
            "tool_uses": tool_uses,
            "tool_responses": [],
            "intermediate_responses": [],
        }
    return turn


def _build_weather_artifact() -> InferenceArtifact:
    """Build a realistic inference artifact for weather agent.

    The agent was asked about London weather:
    - Expected: calls get_weather(city="London")
    - Actual: calls get_weather(city="London") → match!
    """
    eval_case_dict = _make_eval_case_dict(
        eval_id="check_weather__happy_path",
        conversation=[
            _make_conversation_turn(
                inv_id="inv-1",
                user_text="What's the weather in London?",
                response_text="The weather in London is cloudy, 15°C.",
                tool_uses=[{"name": "get_weather", "args": {"city": "London"}}],
            ),
        ],
    )
    eval_set_dict = _make_eval_set_dict(
        eval_set_id="weather_agent__check_weather",
        cases=[eval_case_dict],
    )

    # Build the "actual" invocation (what the agent produced)
    actual_inv = _make_invocation(
        inv_id="inv-1",
        user_text="What's the weather in London?",
        response_text="The weather in London is cloudy, 15°C.",
        tool_calls=[{"name": "get_weather", "args": {"city": "London"}}],
    )
    inference_result = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather_agent__check_weather",
        eval_case_id="check_weather__happy_path",
        session_id="sess-test-001",
        status=InferenceStatus.SUCCESS,
        inferences=[actual_inv],
    )

    return InferenceArtifact(
        eval_set_json=eval_set_dict,
        inference_result_json=json.loads(inference_result.model_dump_json()),
        session_id="sess-test-001",
    )


def _build_mismatched_artifact() -> InferenceArtifact:
    """Build an artifact where agent called wrong tool (should fail).

    Expected: get_weather(city="London")
    Actual: get_forecast(city="London", days=3) → mismatch!
    """
    eval_case_dict = _make_eval_case_dict(
        eval_id="check_weather__wrong_tool",
        conversation=[
            _make_conversation_turn(
                inv_id="inv-1",
                user_text="What's the weather in London?",
                response_text="Here's the forecast for London.",
                tool_uses=[{"name": "get_weather", "args": {"city": "London"}}],
            ),
        ],
    )
    eval_set_dict = _make_eval_set_dict(
        eval_set_id="weather_agent__wrong_tool",
        cases=[eval_case_dict],
    )

    # Actual invocation used wrong tool
    actual_inv = _make_invocation(
        inv_id="inv-1",
        user_text="What's the weather in London?",
        response_text="Here's the forecast for London.",
        tool_calls=[{"name": "get_forecast", "args": {"city": "London", "days": 3}}],
    )
    inference_result = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather_agent__wrong_tool",
        eval_case_id="check_weather__wrong_tool",
        session_id="sess-test-002",
        status=InferenceStatus.SUCCESS,
        inferences=[actual_inv],
    )

    return InferenceArtifact(
        eval_set_json=eval_set_dict,
        inference_result_json=json.loads(inference_result.model_dump_json()),
        session_id="sess-test-002",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferenceResultSerialization:
    """Verify InferenceResult roundtrips through JSON."""

    def test_inference_result_roundtrip(self):
        inv = _make_invocation(
            "inv-1", "hello", "world",
            tool_calls=[{"name": "my_tool", "args": {"x": 1}}],
        )
        ir = InferenceResult(
            app_name="app", eval_set_id="s1", eval_case_id="c1",
            session_id="sess", status=InferenceStatus.SUCCESS,
            inferences=[inv],
        )
        j = ir.model_dump_json()
        ir2 = InferenceResult.model_validate_json(j)
        assert ir2.app_name == "app"
        assert ir2.session_id == "sess"
        assert len(ir2.inferences) == 1
        assert ir2.inferences[0].invocation_id == "inv-1"

    def test_artifact_preserves_inference_result(self):
        artifact = _build_weather_artifact()
        # Reconstruct InferenceResult from stored JSON
        ir = InferenceResult.model_validate(artifact.inference_result_json)
        assert ir.eval_case_id == "check_weather__happy_path"
        assert len(ir.inferences) == 1
        assert ir.inferences[0].invocation_id == "inv-1"


class TestBundlePersistence:
    """Verify bundles save/load to disk correctly."""

    def test_save_and_load_bundle(self, tmp_path):
        bundle = InferenceBundle(
            app_name="test",
            agent_module="examples.weather_agent.agent",
            artifacts=[_build_weather_artifact()],
        )
        path = str(tmp_path / "bundle.json")
        save_inference_bundle(bundle, path)
        loaded = load_inference_bundle(path)
        assert loaded.app_name == "test"
        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].session_id == "sess-test-001"

    def test_bundle_file_is_valid_json(self, tmp_path):
        bundle = InferenceBundle(
            app_name="test",
            artifacts=[_build_weather_artifact()],
        )
        path = tmp_path / "bundle.json"
        save_inference_bundle(bundle, str(path))
        data = json.loads(path.read_text())
        assert "artifacts" in data
        assert len(data["artifacts"]) == 1


class TestEvalMetricBuilding:
    """Verify metric construction from config dicts."""

    def test_build_tool_trajectory_metric(self):
        metrics = build_eval_metrics([
            {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
        ])
        assert len(metrics) == 1
        assert metrics[0].metric_name == "tool_trajectory_avg_score"

    def test_build_multiple_metrics(self):
        metrics = build_eval_metrics([
            {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            {"metric_name": "safety_v1", "threshold": 1.0},
        ])
        assert len(metrics) == 2
        names = {m.metric_name for m in metrics}
        assert "tool_trajectory_avg_score" in names
        assert "safety_v1" in names


class TestEvaluationFromBundle:
    """Core test: evaluate pre-built inference artifacts without agent."""

    @pytest.mark.asyncio
    async def test_evaluate_matching_tools_passes(self):
        """When actual tool calls match expected, score should be 1.0."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_weather_artifact()],
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        r = results[0]
        assert r.status == "PASSED"
        assert r.overall_scores.get("tool_trajectory_avg_score") == 1.0
        assert r.eval_case_id == "check_weather__happy_path"

    @pytest.mark.asyncio
    async def test_evaluate_mismatched_tools_fails(self):
        """When actual tool calls don't match expected, should fail."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_mismatched_artifact()],
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        r = results[0]
        assert r.status == "FAILED"
        score = r.overall_scores.get("tool_trajectory_avg_score")
        assert score is not None and score < 0.8

    @pytest.mark.asyncio
    async def test_evaluate_multiple_artifacts(self):
        """Evaluate a bundle with both passing and failing artifacts."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[
                _build_weather_artifact(),
                _build_mismatched_artifact(),
            ],
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 2
        statuses = {r.eval_case_id: r.status for r in results}
        assert statuses["check_weather__happy_path"] == "PASSED"
        assert statuses["check_weather__wrong_tool"] == "FAILED"

    @pytest.mark.asyncio
    async def test_re_evaluate_with_lower_threshold(self):
        """Same inference, different threshold → different pass/fail."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_mismatched_artifact()],
        )

        # Strict threshold → FAIL
        strict = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert strict[0].status == "FAILED"

        # Lenient threshold → PASS (score >= 0.0)
        lenient = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.0},
            ],
        )
        assert lenient[0].status == "PASSED"


class TestFullRoundtrip:
    """End-to-end: build → save → load → evaluate."""

    @pytest.mark.asyncio
    async def test_save_load_evaluate(self, tmp_path):
        # Build and save
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_weather_artifact()],
        )
        path = str(tmp_path / "inference.json")
        save_inference_bundle(bundle, path)

        # Load and evaluate
        loaded = load_inference_bundle(path)
        results = await run_evaluation_from_bundle(
            loaded,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert len(results) == 1
        assert results[0].status == "PASSED"
        assert results[0].overall_scores["tool_trajectory_avg_score"] == 1.0

    @pytest.mark.asyncio
    async def test_re_evaluate_saved_bundle_with_different_metrics(self, tmp_path):
        """The key value prop: re-score the same inference with different metrics."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_weather_artifact()],
        )
        path = str(tmp_path / "inference.json")
        save_inference_bundle(bundle, path)

        # Evaluate with tool trajectory
        loaded = load_inference_bundle(path)
        r1 = await run_evaluation_from_bundle(
            loaded,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        assert "tool_trajectory_avg_score" in r1[0].overall_scores

        # Re-evaluate same data with safety metric
        loaded2 = load_inference_bundle(path)
        r2 = await run_evaluation_from_bundle(
            loaded2,
            metric_configs=[
                {"metric_name": "safety_v1", "threshold": 1.0},
            ],
        )
        assert "safety_v1" in r2[0].overall_scores

    @pytest.mark.asyncio
    async def test_per_invocation_details(self):
        """Check that per-invocation data is available in results."""
        bundle = InferenceBundle(
            app_name="eval_app",
            artifacts=[_build_weather_artifact()],
        )
        results = await run_evaluation_from_bundle(
            bundle,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        r = results[0]
        assert len(r.per_invocation) == 1
        inv = r.per_invocation[0]
        assert inv["invocation_id"] == "inv-1"
        assert "What's the weather" in inv["user_message"]
        assert inv["actual_tool_calls"][0]["name"] == "get_weather"
        assert inv["expected_tool_calls"][0]["name"] == "get_weather"
        assert "tool_trajectory_avg_score" in inv["scores"]
