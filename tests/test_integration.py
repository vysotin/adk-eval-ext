"""Integration tests for the full pipeline (no LLM calls)."""

import json
from pathlib import Path

import pytest
from google.adk.agents import Agent

from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.schemas import (
    AgentMetadata,
    TaskScenarioSet,
    Task,
    Scenario,
    TestCaseConfig,
)
from adk_eval_tool.testcase_generator.tools import build_eval_set_json, validate_eval_set
from adk_eval_tool.task_generator.tools import validate_task_output


def _dummy_search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"


def test_full_pipeline_parse_to_evalset(tmp_path):
    """Test the full pipeline: parse agent -> build tasks -> generate evalset."""
    agent = Agent(
        name="test_agent", model="gemini-2.0-flash",
        description="A test agent", instruction="Help users search.",
        tools=[_dummy_search],
    )
    metadata = parse_agent(agent, save_path=str(tmp_path / "metadata.json"))
    assert metadata.name == "test_agent"

    task_set = TaskScenarioSet(
        agent_name=metadata.name,
        tasks=[Task(
            task_id="search_info", name="Search Information",
            description="User wants to search for information",
            scenarios=[
                Scenario(scenario_id="basic_search", name="Basic search",
                    description="User searches for Python tutorials, expects relevant results"),
                Scenario(scenario_id="empty_query", name="Empty query",
                    description="User provides empty query, agent should ask for clarification"),
            ],
        )],
    )
    validation = validate_task_output(task_set.model_dump())
    assert validation["valid"] is True

    for task in task_set.tasks:
        eval_set = build_eval_set_json(task.model_dump(), metadata.name)
        result = validate_eval_set(json.dumps(eval_set))
        assert result["valid"] is True
        assert eval_set["evalSetId"] == "test_agent__search_info"
        assert len(eval_set["evalCases"]) == 2


def test_metadata_roundtrip_through_json(tmp_path):
    agent = Agent(
        name="roundtrip_agent", model="gemini-2.0-flash",
        description="Tests roundtrip", instruction="You are a test.",
        tools=[_dummy_search],
        sub_agents=[Agent(name="sub_agent", model="gemini-2.0-flash", instruction="Sub.")],
    )
    metadata = parse_agent(agent)
    restored = AgentMetadata.model_validate_json(metadata.model_dump_json())
    assert restored.name == metadata.name
    assert len(restored.sub_agents) == 1


def test_trace_tree_from_spans():
    from adk_eval_tool.eval_runner.trace_collector import build_trace_tree, SpanData

    spans = [
        SpanData(span_id="s1", name="invocation", start_time=0, end_time=10),
        SpanData(span_id="s2", name="call_llm", parent_span_id="s1", start_time=1, end_time=3),
        SpanData(span_id="s3", name="execute_tool:search", parent_span_id="s1", start_time=3, end_time=5),
    ]
    tree = build_trace_tree(spans)
    assert len(tree) == 1
    assert len(tree[0].children) == 2


def test_result_store_full_workflow(tmp_path):
    from adk_eval_tool.eval_runner.result_store import ResultStore
    from adk_eval_tool.schemas import EvalRunResult

    store = ResultStore(base_dir=str(tmp_path / "results"))
    for i, (score, status) in enumerate([(0.9, "PASSED"), (0.7, "FAILED"), (0.85, "PASSED")]):
        store.save_result(EvalRunResult(
            run_id=f"run-{i}", eval_set_id="test_set", eval_id=f"case_{i}",
            status=status, overall_scores={"tool_trajectory_avg_score": score}, timestamp=float(i)))

    assert len(store.load_results()) == 3
    avgs = store.compute_averages(eval_set_id="test_set")
    assert abs(avgs["tool_trajectory_avg_score"] - 0.8167) < 0.01


def test_eval_run_config_to_adk_config():
    from adk_eval_tool.schemas import EvalRunConfig, MetricConfig
    from adk_eval_tool.eval_runner.runner import _build_eval_config_from_metrics

    config = EvalRunConfig(
        agent_module="test_agent",
        metrics=[
            MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.9, match_type="IN_ORDER"),
            MetricConfig(metric_name="hallucinations_v1", threshold=0.8, evaluate_intermediate=True),
        ],
        judge_model="gemini-2.5-flash",
    )
    eval_config = _build_eval_config_from_metrics(config.metrics, config.judge_model)
    assert "tool_trajectory_avg_score" in eval_config.criteria
    assert "hallucinations_v1" in eval_config.criteria
