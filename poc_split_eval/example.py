"""Example: split evaluation into inference + evaluation stages.

This example demonstrates the two-stage workflow:

Stage 1 — Inference (requires agent + LLM):
    Run the weather agent against eval cases, persist results to disk.

Stage 2 — Evaluation (no agent needed):
    Load persisted results, score them with configurable metrics.
    Can be re-run with different metrics without re-running the agent.

Usage:
    # Run both stages
    python -m poc_split_eval.example

    # Or import and use programmatically
    from poc_split_eval.example import run_example
    import asyncio
    asyncio.run(run_example())
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from google.adk.evaluation.base_eval_service import InferenceResult, InferenceStatus
from google.adk.evaluation.eval_case import Invocation, IntermediateData
from google.genai import types as genai_types

from poc_split_eval.schemas import InferenceArtifact, InferenceBundle
from poc_split_eval.inference import save_inference_bundle, load_inference_bundle
from poc_split_eval.evaluation import run_evaluation_from_bundle


# ---------------------------------------------------------------------------
# Simulate Stage 1 output (without actually calling the LLM)
# ---------------------------------------------------------------------------

def _build_simulated_bundle() -> InferenceBundle:
    """Build a bundle that simulates what Stage 1 would produce.

    In production, you'd use `run_inference()` from inference.py.
    Here we build realistic artifacts by hand so the example runs
    without API keys.
    """
    # The eval set: what we expected the agent to do
    eval_set = {
        "eval_set_id": "weather_agent__check_weather",
        "eval_cases": [
            {
                "eval_id": "happy_path",
                "conversation": [
                    {
                        "invocation_id": "inv-1",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "What's the weather in London?"}],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [{"text": "The weather in London is cloudy, 15°C."}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {"name": "get_weather", "args": {"city": "London"}}
                            ],
                            "tool_responses": [],
                            "intermediate_responses": [],
                        },
                    },
                ],
            },
            {
                "eval_id": "wrong_tool_used",
                "conversation": [
                    {
                        "invocation_id": "inv-2",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "What's the weather in Tokyo?"}],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [{"text": "Weather in Tokyo is humid, 28°C."}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {"name": "get_weather", "args": {"city": "Tokyo"}}
                            ],
                            "tool_responses": [],
                            "intermediate_responses": [],
                        },
                    },
                ],
            },
        ],
    }

    # Simulated inference results — what the agent actually did
    # Case 1: agent called the right tool → should PASS
    correct_inv = Invocation(
        invocation_id="inv-1",
        user_content=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="What's the weather in London?")],
        ),
        final_response=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="The weather in London is cloudy, 15°C.")],
        ),
        intermediate_data=IntermediateData(
            tool_uses=[
                genai_types.FunctionCall(
                    name="get_weather", args={"city": "London"}
                ),
            ],
            tool_responses=[],
            intermediate_responses=[],
        ),
    )
    ir_correct = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather_agent__check_weather",
        eval_case_id="happy_path",
        session_id="sess-001",
        status=InferenceStatus.SUCCESS,
        inferences=[correct_inv],
    )

    # Case 2: agent called get_forecast instead of get_weather → should FAIL
    wrong_inv = Invocation(
        invocation_id="inv-2",
        user_content=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="What's the weather in Tokyo?")],
        ),
        final_response=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="Here's the 3-day forecast for Tokyo.")],
        ),
        intermediate_data=IntermediateData(
            tool_uses=[
                genai_types.FunctionCall(
                    name="get_forecast",
                    args={"city": "Tokyo", "days": 3},
                ),
            ],
            tool_responses=[],
            intermediate_responses=[],
        ),
    )
    ir_wrong = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather_agent__check_weather",
        eval_case_id="wrong_tool_used",
        session_id="sess-002",
        status=InferenceStatus.SUCCESS,
        inferences=[wrong_inv],
    )

    return InferenceBundle(
        app_name="eval_app",
        agent_module="examples.weather_agent.agent",
        artifacts=[
            InferenceArtifact(
                eval_set_json=eval_set,
                inference_result_json=json.loads(ir_correct.model_dump_json()),
                session_id="sess-001",
            ),
            InferenceArtifact(
                eval_set_json=eval_set,
                inference_result_json=json.loads(ir_wrong.model_dump_json()),
                session_id="sess-002",
            ),
        ],
        metadata={"simulated": True},
    )


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

async def run_example():
    print("=" * 60)
    print("POC: Split ADK Evaluation into Inference + Evaluation")
    print("=" * 60)

    # --- Stage 1: Build inference artifacts ---
    print("\n--- Stage 1: Inference (simulated) ---")
    bundle = _build_simulated_bundle()
    print(f"  Built {len(bundle.artifacts)} inference artifacts")

    # Save to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = str(Path(tmpdir) / "inference_bundle.json")
        save_inference_bundle(bundle, bundle_path)
        file_size = Path(bundle_path).stat().st_size
        print(f"  Saved to: {bundle_path} ({file_size:,} bytes)")

        # --- Stage 2a: Evaluate with tool trajectory metric ---
        print("\n--- Stage 2a: Evaluate with tool_trajectory (threshold=0.8) ---")
        loaded = load_inference_bundle(bundle_path)
        results = await run_evaluation_from_bundle(
            loaded,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        for r in results:
            print(f"  {r.eval_case_id}: {r.status} "
                  f"(score={r.overall_scores.get('tool_trajectory_avg_score')})")

        # --- Stage 2b: Re-evaluate with lenient threshold ---
        print("\n--- Stage 2b: Re-evaluate same data with threshold=0.0 ---")
        loaded2 = load_inference_bundle(bundle_path)
        results2 = await run_evaluation_from_bundle(
            loaded2,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.0},
            ],
        )
        for r in results2:
            print(f"  {r.eval_case_id}: {r.status} "
                  f"(score={r.overall_scores.get('tool_trajectory_avg_score')})")

        # --- Stage 2c: Evaluate with different metrics entirely ---
        print("\n--- Stage 2c: Evaluate same data with safety_v1 ---")
        loaded3 = load_inference_bundle(bundle_path)
        results3 = await run_evaluation_from_bundle(
            loaded3,
            metric_configs=[
                {"metric_name": "safety_v1", "threshold": 1.0},
            ],
        )
        for r in results3:
            print(f"  {r.eval_case_id}: {r.status} "
                  f"(scores={r.overall_scores})")

    # === Part 2: Mocked tools demonstration ===
    print("\n" + "=" * 60)
    print("Part 2: Inference with MOCKED TOOLS")
    print("=" * 60)

    # Eval set with explicit tool_responses for mocking
    mock_eval_set = {
        "eval_set_id": "weather_agent__check_weather",
        "eval_cases": [
            {
                "eval_id": "happy_path",
                "conversation": [
                    {
                        "invocation_id": "inv-1",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "What's the weather in London?"}],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [{"text": "The weather in London is cloudy, 15°C."}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {"name": "get_weather", "args": {"city": "London"}}
                            ],
                            "tool_responses": [
                                {
                                    "name": "get_weather",
                                    "response": {"city": "London", "temp_c": 15, "condition": "Cloudy"},
                                }
                            ],
                            "intermediate_responses": [],
                        },
                    },
                ],
            },
        ],
    }

    print("\n--- Build tool response map from eval set ---")
    from poc_split_eval.mocking import build_tool_response_map

    response_map = build_tool_response_map(mock_eval_set)
    for tool_name, response in response_map.items():
        print(f"  {tool_name} → {json.dumps(response, default=str)[:80]}")

    print("\n--- Build mocked inference artifacts ---")
    # Simulate an agent that used get_weather with the mocked response
    mocked_inv = Invocation(
        invocation_id="inv-1",
        user_content=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="What's the weather in London?")],
        ),
        final_response=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="London is 15°C and cloudy (from mock).")],
        ),
        intermediate_data=IntermediateData(
            tool_uses=[
                genai_types.FunctionCall(
                    name="get_weather", args={"city": "London"}
                ),
            ],
            tool_responses=[],
            intermediate_responses=[],
        ),
    )
    ir_mocked = InferenceResult(
        app_name="eval_app",
        eval_set_id="weather_agent__check_weather",
        eval_case_id="happy_path",
        session_id="sess-mocked-001",
        status=InferenceStatus.SUCCESS,
        inferences=[mocked_inv],
    )

    mocked_bundle = InferenceBundle(
        app_name="eval_app",
        artifacts=[
            InferenceArtifact(
                eval_set_json=mock_eval_set,
                inference_result_json=json.loads(ir_mocked.model_dump_json()),
                session_id="sess-mocked-001",
            ),
        ],
        metadata={"mock_tools": True, "tool_response_map": response_map},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mocked_path = str(Path(tmpdir) / "mocked_bundle.json")
        save_inference_bundle(mocked_bundle, mocked_path)
        size = Path(mocked_path).stat().st_size
        print(f"  Saved mocked bundle: {size:,} bytes")

        print("\n--- Evaluate mocked inference ---")
        loaded = load_inference_bundle(mocked_path)
        results = await run_evaluation_from_bundle(
            loaded,
            metric_configs=[
                {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
            ],
        )
        for r in results:
            print(f"  {r.eval_case_id}: {r.status} "
                  f"(score={r.overall_scores.get('tool_trajectory_avg_score')})")

    # === Part 3: Stubbed sub-agents demonstration ===
    print("\n" + "=" * 60)
    print("Part 3: Inference with STUBBED SUB-AGENTS")
    print("=" * 60)

    travel_eval_set = {
        "eval_set_id": "travel__stubbed",
        "eval_cases": [
            {
                "eval_id": "flight_stubbed",
                "conversation": [
                    {
                        "invocation_id": "inv-1",
                        "user_content": {
                            "role": "user",
                            "parts": [{"text": "Find flights London to Paris"}],
                        },
                        "intermediate_data": {
                            "tool_uses": [
                                {"name": "search_flights",
                                 "args": {"origin": "London", "destination": "Paris"}},
                            ],
                            "tool_responses": [
                                {"name": "search_flights",
                                 "response": {"flights": [{"id": "FL-101"}]}},
                            ],
                            "intermediate_responses": [],
                        },
                        "final_response": {
                            "role": "model",
                            "parts": [{"text": "Found flight FL-101."}],
                        },
                    },
                ],
            },
        ],
    }

    stubbed_inv = Invocation(
        invocation_id="inv-1",
        user_content=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="Find flights London to Paris")],
        ),
        final_response=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="Found flight FL-101 (via stub).")],
        ),
        intermediate_data=IntermediateData(
            tool_uses=[
                genai_types.FunctionCall(
                    name="search_flights",
                    args={"origin": "London", "destination": "Paris"},
                ),
            ],
            tool_responses=[],
            intermediate_responses=[],
        ),
    )
    ir_stubbed = InferenceResult(
        app_name="eval_app",
        eval_set_id="travel__stubbed",
        eval_case_id="flight_stubbed",
        session_id="sess-stubbed-001",
        status=InferenceStatus.SUCCESS,
        inferences=[stubbed_inv],
    )

    stubbed_bundle = InferenceBundle(
        app_name="eval_app",
        artifacts=[
            InferenceArtifact(
                eval_set_json=travel_eval_set,
                inference_result_json=json.loads(ir_stubbed.model_dump_json()),
                session_id="sess-stubbed-001",
            ),
        ],
        metadata={
            "stub_sub_agents": {"flight_agent": "Stub: flight search results"},
        },
    )

    print("  Sub-agents stubbed: flight_agent")
    results = await run_evaluation_from_bundle(
        stubbed_bundle,
        metric_configs=[
            {"metric_name": "tool_trajectory_avg_score", "threshold": 0.8},
        ],
    )
    for r in results:
        print(f"  {r.eval_case_id}: {r.status} "
              f"(score={r.overall_scores.get('tool_trajectory_avg_score')})")

    print("\n" + "=" * 60)
    print("Summary:")
    print("  Part 1: Basic split eval (inference once, evaluate many times)")
    print("  Part 2: Mocked tools (canned responses from eval set data)")
    print("  Part 3: Stubbed sub-agents (lightweight stubs replace real agents)")
    print("  All three patterns persist to JSON and evaluate without agents!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_example())
