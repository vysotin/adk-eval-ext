"""Integration tests for the full pipeline (no LLM calls)."""

import json
from pathlib import Path

import pytest
from google.adk.agents import Agent

from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.schemas import (
    AgentMetadata,
    IntentScenarioSet,
    Intent,
    Scenario,
    ScenarioStep,
    TestCaseConfig,
)
from adk_eval_tool.testcase_generator.tools import build_eval_set_json, validate_eval_set
from adk_eval_tool.intent_generator.tools import validate_intent_output


def _dummy_search(query: str) -> str:
    """Search for information.

    Args:
        query: Search query.

    Returns:
        Search results.
    """
    return f"Results for {query}"


def test_full_pipeline_parse_to_evalset(tmp_path):
    """Test the full pipeline: parse agent -> build intents -> generate evalset."""
    # Step 1: Parse agent
    agent = Agent(
        name="test_agent",
        model="gemini-2.0-flash",
        description="A test agent",
        instruction="Help users search for information.",
        tools=[_dummy_search],
    )
    metadata = parse_agent(agent, save_path=str(tmp_path / "metadata.json"))
    assert metadata.name == "test_agent"
    assert (tmp_path / "metadata.json").exists()

    # Step 2: Create intents manually (simulating LLM output)
    intent_set = IntentScenarioSet(
        agent_name=metadata.name,
        intents=[
            Intent(
                intent_id="search_info",
                name="Search Information",
                description="User wants to search for information",
                category="search",
                scenarios=[
                    Scenario(
                        scenario_id="basic_search",
                        name="Basic search query",
                        steps=[
                            ScenarioStep(
                                user_message="Search for Python tutorials",
                                expected_tool_calls=["_dummy_search"],
                                expected_response_keywords=["Python", "tutorial"],
                            ),
                        ],
                        tags=["happy_path"],
                    ),
                    Scenario(
                        scenario_id="empty_query",
                        name="Empty search query",
                        steps=[
                            ScenarioStep(
                                user_message="Search for",
                                expected_tool_calls=[],
                                notes="Agent should ask for clarification",
                            ),
                        ],
                        tags=["edge_case"],
                    ),
                ],
            ),
        ],
    )
    validation = validate_intent_output(intent_set.model_dump())
    assert validation["valid"] is True

    # Step 3: Generate eval set
    for intent in intent_set.intents:
        eval_set = build_eval_set_json(intent.model_dump(), metadata.name)
        eval_json = json.dumps(eval_set)
        result = validate_eval_set(eval_json)
        assert result["valid"] is True, f"Invalid eval set: {result['errors']}"
        assert eval_set["evalSetId"] == "test_agent__search_info"
        assert len(eval_set["evalCases"]) == 2

    # Step 4: Save eval set
    eval_path = tmp_path / "evalsets"
    eval_path.mkdir()
    for intent in intent_set.intents:
        eval_set = build_eval_set_json(intent.model_dump(), metadata.name)
        filepath = eval_path / f"{eval_set['evalSetId']}.evalset.json"
        filepath.write_text(json.dumps(eval_set, indent=2))
        assert filepath.exists()


def test_metadata_roundtrip_through_json(tmp_path):
    """Test that metadata survives JSON serialization/deserialization."""
    agent = Agent(
        name="roundtrip_agent",
        model="gemini-2.0-flash",
        description="Tests roundtrip",
        instruction="You are a test.",
        tools=[_dummy_search],
        sub_agents=[
            Agent(
                name="sub_agent",
                model="gemini-2.0-flash",
                instruction="Sub task.",
                tools=[],
            )
        ],
    )
    metadata = parse_agent(agent)
    json_str = metadata.model_dump_json(indent=2)
    restored = AgentMetadata.model_validate_json(json_str)

    assert restored.name == metadata.name
    assert len(restored.tools) == len(metadata.tools)
    assert len(restored.sub_agents) == len(metadata.sub_agents)
    assert restored.sub_agents[0].name == "sub_agent"


def test_trace_tree_from_spans():
    """Test building a trace tree from flat spans."""
    from adk_eval_tool.eval_runner.trace_collector import build_trace_tree, SpanData

    spans = [
        SpanData(span_id="s1", name="invocation", start_time=0, end_time=10),
        SpanData(span_id="s2", name="call_llm", parent_span_id="s1", start_time=1, end_time=3),
        SpanData(span_id="s3", name="execute_tool:search", parent_span_id="s1", start_time=3, end_time=5),
        SpanData(span_id="s4", name="call_llm", parent_span_id="s1", start_time=5, end_time=8),
    ]
    tree = build_trace_tree(spans)

    assert len(tree) == 1
    root = tree[0]
    assert root.name == "invocation"
    assert len(root.children) == 3
    assert root.children[1].name == "execute_tool:search"


def test_result_store_full_workflow(tmp_path):
    """Test result store save/load/averages workflow."""
    from adk_eval_tool.eval_runner.result_store import ResultStore
    from adk_eval_tool.schemas import EvalRunResult

    store = ResultStore(base_dir=str(tmp_path / "results"))

    for i, (score, status) in enumerate([(0.9, "PASSED"), (0.7, "FAILED"), (0.85, "PASSED")]):
        store.save_result(EvalRunResult(
            run_id=f"run-{i}",
            eval_set_id="test_set",
            eval_id=f"case_{i}",
            status=status,
            overall_scores={"tool_trajectory_avg_score": score, "safety_v1": 1.0},
            timestamp=float(i),
        ))

    all_results = store.load_results()
    assert len(all_results) == 3

    filtered = store.load_results(eval_set_id="test_set")
    assert len(filtered) == 3

    avgs = store.compute_averages(eval_set_id="test_set")
    assert abs(avgs["tool_trajectory_avg_score"] - 0.8167) < 0.01
    assert avgs["safety_v1"] == 1.0


def test_eval_run_config_to_adk_config():
    """Test that EvalRunConfig can be converted to ADK EvalConfig."""
    from adk_eval_tool.schemas import EvalRunConfig, MetricConfig
    from adk_eval_tool.eval_runner.runner import _build_eval_config_from_metrics

    config = EvalRunConfig(
        agent_module="test_agent",
        metrics=[
            MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.9, match_type="IN_ORDER"),
            MetricConfig(metric_name="safety_v1", threshold=1.0),
            MetricConfig(metric_name="hallucinations_v1", threshold=0.8, evaluate_intermediate=True),
        ],
        judge_model="gemini-2.5-flash",
    )

    eval_config = _build_eval_config_from_metrics(config.metrics, config.judge_model)
    assert "tool_trajectory_avg_score" in eval_config.criteria
    assert "safety_v1" in eval_config.criteria
    assert "hallucinations_v1" in eval_config.criteria
