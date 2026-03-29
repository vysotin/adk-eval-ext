"""End-to-end test for the travel_multi_agent through all UI steps.

Pages: Agent Metadata, Tasks & Trajectories, Test Cases, Inference, Evaluation
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from adk_eval_tool.agent_parser.parser import parse_agent_from_source
from adk_eval_tool.schemas import (
    AgentMetadata, ToolMetadata, TaskScenarioSet, Task, Scenario,
    EvalRunConfig, MetricConfig, InferenceRunResult, EvalRunResult, BasicMetrics, TraceSpanNode,
)


APP_FILE = str(Path(__file__).resolve().parent.parent.parent / "adk_eval_tool" / "ui" / "app.py")
TRAVEL_AGENT_SRC = str(Path(__file__).resolve().parent.parent.parent / "examples" / "travel_multi_agent" / "agent.py")

PAGES = [
    "Agent Metadata",
    "Tasks & Trajectories",
    "Test Cases",
    "Inference",
    "Evaluation",
]

# Only non-GCP metrics (no safety_v1, no per_turn_user_simulator_quality_v1)
ALL_METRICS = [
    MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8, match_type="IN_ORDER"),
    MetricConfig(metric_name="response_match_score", threshold=0.8),
    MetricConfig(metric_name="response_evaluation_score", threshold=0.7),
    MetricConfig(metric_name="final_response_match_v2", threshold=0.8),
    MetricConfig(metric_name="hallucinations_v1", threshold=0.9, evaluate_intermediate=True),
]


def _travel_metadata() -> AgentMetadata:
    return parse_agent_from_source(TRAVEL_AGENT_SRC)

def _travel_task_set() -> TaskScenarioSet:
    return TaskScenarioSet(
        agent_name="travel_coordinator",
        tasks=[
            Task(task_id="search_flights", name="Search Flights",
                description="Find flights", trajectories=[
                    Scenario(scenario_id="flight_happy_path", name="Flight search",
                        description="User provides origin, destination, and date; expects list of flights")]),
        ],
    )

def _travel_eval_sets() -> list[dict]:
    return [{
        "evalSetId": "travel_coordinator__search_flights", "name": "Search Flights",
        "evalCases": [{"evalId": "flight_happy_path", "conversation": [{
            "invocationId": "inv-1",
            "userContent": {"role": "user", "parts": [{"text": "Find flights from London to Paris"}]},
            "finalResponse": {"role": "model", "parts": [{"text": "Found 2 flights."}]},
            "intermediateData": {
                "toolUses": [{"name": "search_flights", "args": {"origin": "London", "destination": "Paris", "date": "2025-06-15"}}],
                "toolResponses": [{"name": "search_flights", "id": "c1", "response": {"flights": [{"id": "FL-101"}]}}],
                "intermediateResponses": [],
            },
        }]}],
    }]

def _travel_inference_results() -> list[InferenceRunResult]:
    return [InferenceRunResult(
        run_id="inf-travel-001", eval_set_id="travel_coordinator__search_flights",
        eval_id="flight_happy_path", session_id="sess-travel-001",
        actual_invocations=[{"invocation_id": "inv-1",
            "user_message": "Find flights from London to Paris",
            "actual_response": "Found 2 flights.",
            "actual_tool_calls": [{"name": "search_flights", "args": {"origin": "London", "destination": "Paris"}}]}],
        basic_metrics=BasicMetrics(total_input_tokens=800, total_output_tokens=150, total_tokens=950,
            num_llm_calls=2, num_tool_calls=1, total_duration_ms=2500.0, max_context_size=800),
        trace_tree=TraceSpanNode(span_id="root-1", name="invocation", start_time=0.0, end_time=2.5, children=[
            TraceSpanNode(span_id="llm-1", name="call_llm", start_time=0.0, end_time=0.8,
                attributes={"gen_ai.usage.input_tokens": "800", "gen_ai.usage.output_tokens": "150"}),
            TraceSpanNode(span_id="tool-1", name="execute_tool:search_flights", start_time=0.8, end_time=1.2),
        ]),
        timestamp=1000.0,
    )]

def _travel_eval_results() -> list[EvalRunResult]:
    scores = {"tool_trajectory_avg_score": 1.0, "response_match_score": 0.85}
    return [EvalRunResult(
        run_id="run-travel-001", eval_set_id="travel_coordinator__search_flights",
        eval_id="flight_happy_path", status="PASSED", overall_scores=scores,
        per_invocation_scores=[{"invocation_id": "inv-1",
            "user_message": "Find flights from London to Paris",
            "actual_response": "Found 2 flights.", "expected_response": "Found 2 flights.",
            "actual_tool_calls": [{"name": "search_flights", "args": {"origin": "London", "destination": "Paris"}}],
            "expected_tool_calls": [{"name": "search_flights", "args": {"origin": "London", "destination": "Paris"}}],
            "scores": scores}],
        basic_metrics=BasicMetrics(total_input_tokens=800, total_output_tokens=150, total_tokens=950,
            num_llm_calls=2, num_tool_calls=1, total_duration_ms=2500.0, max_context_size=800),
        session_id="sess-travel-001", timestamp=1000.0,
    )]

def _preload_travel_agent(metadata: AgentMetadata):
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "metadata.json"
    path.write_text(metadata.model_dump_json(indent=2))
    os.environ["ADK_EVAL_PRELOADED_METADATA"] = str(path)
    os.environ["ADK_EVAL_AGENT_MODULE"] = "examples.travel_multi_agent.agent"
    os.environ["ADK_EVAL_AGENT_VARIABLE"] = "root_agent"

def _navigate_to_step(at: AppTest, step_idx: int) -> AppTest:
    at.session_state["current_step"] = step_idx
    return at.run()

def _navigate_to_page(at: AppTest, page_name: str) -> AppTest:
    return _navigate_to_step(at, PAGES.index(page_name))


class TestTravelAgentMetadataParsing:
    def test_parse_travel_agent(self):
        meta = _travel_metadata()
        assert meta.name == "travel_coordinator"
        assert len(meta.sub_agents) == 2

    def test_sub_agents_have_tools(self):
        meta = _travel_metadata()
        flight = next(s for s in meta.sub_agents if s.name == "flight_agent")
        hotel = next(s for s in meta.sub_agents if s.name == "hotel_agent")
        assert flight.tools[0].name == "search_flights"
        assert hotel.tools[0].name == "search_hotels"


class TestTravelStep0:
    def test_loads(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert not at.exception
        sidebar_texts = [el.value for el in at.sidebar.success]
        assert any("travel_coordinator" in t for t in sidebar_texts)

    def test_shows_sub_agents_info(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE).run()
        sidebar_captions = [el.value for el in at.sidebar.caption]
        assert any("2 sub-agents" in c for c in sidebar_captions)


class TestTravelStep1Tasks:
    def test_navigate(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE).run()
        at = _navigate_to_page(at, "Tasks & Trajectories")
        assert not at.exception


class TestTravelStep2TestCases:
    def test_navigate(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _travel_task_set()
        at.run()
        at = _navigate_to_page(at, "Test Cases")
        assert not at.exception


class TestTravelStep3Inference:
    def test_navigate(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _travel_task_set()
        at.session_state["eval_sets"] = _travel_eval_sets()
        at.run()
        at = _navigate_to_page(at, "Inference")
        assert not at.exception
        headers = [el.value for el in at.header]
        assert any("Inference" in h for h in headers)

    def test_agent_module_prefilled(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _travel_task_set()
        at.session_state["eval_sets"] = _travel_eval_sets()
        at.run()
        at = _navigate_to_page(at, "Inference")
        module_input = at.text_input(key="eval_agent_module")
        assert module_input.value == "examples.travel_multi_agent.agent"


class TestTravelStep4Evaluation:
    def test_navigate(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _travel_task_set()
        at.session_state["eval_sets"] = _travel_eval_sets()
        at.session_state["inference_results"] = _travel_inference_results()
        at.run()
        at = _navigate_to_page(at, "Evaluation")
        assert not at.exception
        headers = [el.value for el in at.header]
        assert any("Evaluation" in h for h in headers)

    def test_metric_checkboxes_no_gcp(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _travel_task_set()
        at.session_state["eval_sets"] = _travel_eval_sets()
        at.session_state["inference_results"] = _travel_inference_results()
        at.run()
        at = _navigate_to_page(at, "Evaluation")
        labels = [cb.label for cb in at.checkbox]
        assert "tool_trajectory_avg_score" in labels
        assert "hallucinations_v1" in labels
        # GCP-dependent metrics excluded
        assert "safety_v1" not in labels
        assert "per_turn_user_simulator_quality_v1" not in labels


class TestTravelFullFlow:
    def test_all_pages_no_exceptions(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _travel_task_set()
        at.session_state["eval_sets"] = _travel_eval_sets()
        at.session_state["inference_results"] = _travel_inference_results()
        at.session_state["eval_run_config"] = EvalRunConfig(
            agent_module="examples.travel_multi_agent.agent", metrics=ALL_METRICS, judge_model="gemini-2.5-flash")
        at.session_state["eval_results"] = _travel_eval_results()
        at.run()
        assert not at.exception
        for i, page in enumerate(PAGES):
            at = _navigate_to_step(at, i)
            assert not at.exception, f"Exception on page '{page}': {at.exception}"

    def test_step_unlocking_sequence(self):
        _preload_travel_agent(_travel_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert not at.button(key="nav_step_0").disabled
        assert not at.button(key="nav_step_1").disabled
        assert at.button(key="nav_step_2").disabled

        at.session_state["task_set"] = _travel_task_set()
        at = at.run()
        assert not at.button(key="nav_step_2").disabled
        assert at.button(key="nav_step_3").disabled

        at.session_state["eval_sets"] = _travel_eval_sets()
        at = at.run()
        assert not at.button(key="nav_step_3").disabled
        assert at.button(key="nav_step_4").disabled

        at.session_state["inference_results"] = _travel_inference_results()
        at = at.run()
        assert not at.button(key="nav_step_4").disabled
