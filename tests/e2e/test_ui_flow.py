"""End-to-end tests for the Streamlit UI flow using AppTest.

Pages: Agent Metadata, Tasks & Trajectories, Test Cases, Inference, Evaluation
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    TaskScenarioSet,
    Task,
    Scenario,
    EvalRunConfig,
    MetricConfig,
    InferenceRunResult,
    EvalRunResult,
    BasicMetrics,
    TraceSpanNode,
)


PAGES = [
    "Agent Metadata",
    "Tasks & Trajectories",
    "Test Cases",
    "Inference",
    "Evaluation",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="weather_agent", agent_type="LlmAgent",
        description="A weather assistant", instruction="You help users check weather.",
        model="gemini-2.0-flash",
        tools=[
            ToolMetadata(name="get_weather", description="Get current weather", parameters_schema={
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        ],
        sub_agents=[],
    )

def _sample_task_set() -> TaskScenarioSet:
    return TaskScenarioSet(
        agent_name="weather_agent",
        tasks=[Task(task_id="check_weather", name="Check Weather",
            description="Check weather", trajectories=[
                Scenario(scenario_id="happy_path", name="Happy path",
                    description="User asks for weather in a known city, expects temperature")])],
    )

def _sample_eval_sets() -> list[dict]:
    return [{
        "evalSetId": "weather_agent__check_weather", "name": "Check Weather",
        "evalCases": [{"evalId": "happy_path", "conversation": [{
            "invocationId": "inv-1",
            "userContent": {"role": "user", "parts": [{"text": "Weather in London?"}]},
            "finalResponse": {"role": "model", "parts": [{"text": "Cloudy, 15C."}]},
            "intermediateData": {"toolUses": [{"name": "get_weather", "args": {"city": "London"}}],
                "toolResponses": [], "intermediateResponses": []},
        }]}],
    }]

def _sample_inference_results() -> list[InferenceRunResult]:
    return [InferenceRunResult(
        run_id="inf-001", eval_set_id="weather_agent__check_weather",
        eval_id="happy_path", session_id="sess-001",
        actual_invocations=[{"invocation_id": "inv-1", "user_message": "Weather in London?",
            "actual_response": "Cloudy, 15C.", "actual_tool_calls": [{"name": "get_weather", "args": {"city": "London"}}]}],
        basic_metrics=BasicMetrics(total_input_tokens=500, total_output_tokens=80, total_tokens=580,
            num_llm_calls=1, num_tool_calls=1, total_duration_ms=1200.0, max_context_size=500),
        timestamp=1000.0,
    )]

def _sample_eval_results() -> list[EvalRunResult]:
    return [EvalRunResult(
        run_id="run-001", eval_set_id="weather_agent__check_weather",
        eval_id="happy_path", status="PASSED",
        overall_scores={"tool_trajectory_avg_score": 1.0},
        per_invocation_scores=[{"invocation_id": "inv-1", "user_message": "Weather in London?",
            "actual_response": "Cloudy, 15C.", "expected_response": "Cloudy, 15C.",
            "actual_tool_calls": [{"name": "get_weather", "args": {"city": "London"}}],
            "expected_tool_calls": [{"name": "get_weather", "args": {"city": "London"}}],
            "scores": {"tool_trajectory_avg_score": 1.0}}],
        basic_metrics=BasicMetrics(total_input_tokens=500, total_output_tokens=80, total_tokens=580,
            num_llm_calls=1, num_tool_calls=1, total_duration_ms=1200.0, max_context_size=500),
        session_id="sess-001", timestamp=1000.0,
    )]

def _preload_metadata_env(metadata: AgentMetadata):
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "metadata.json"
    path.write_text(metadata.model_dump_json(indent=2))
    os.environ["ADK_EVAL_PRELOADED_METADATA"] = str(path)
    os.environ["ADK_EVAL_AGENT_MODULE"] = "examples.weather_agent.agent"
    os.environ["ADK_EVAL_AGENT_VARIABLE"] = "root_agent"

def _navigate_to_step(at: AppTest, step_idx: int) -> AppTest:
    at.session_state["current_step"] = step_idx
    return at.run()

def _navigate_to_page(at: AppTest, page_name: str) -> AppTest:
    return _navigate_to_step(at, PAGES.index(page_name))


APP_FILE = str(Path(__file__).resolve().parent.parent.parent / "adk_eval_tool" / "ui" / "app.py")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentMetadataPage:
    def test_app_loads_with_preloaded_metadata(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert not at.exception, f"Exception: {at.exception}"
        sidebar_texts = [el.value for el in at.sidebar.success]
        assert any("weather_agent" in t for t in sidebar_texts)

    def test_metadata_page_shows_header(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        headers = [el.value for el in at.header]
        assert any("Agent Metadata" in h for h in headers)


class TestTasksPage:
    def test_navigate(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        at = _navigate_to_page(at, "Tasks & Trajectories")
        assert not at.exception

    def test_shows_tasks_when_preloaded(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.run()
        at = _navigate_to_page(at, "Tasks & Trajectories")
        assert not at.exception


class TestTestCasesPage:
    def test_navigate(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.run()
        at = _navigate_to_page(at, "Test Cases")
        assert not at.exception
        headers = [el.value for el in at.header]
        assert any("Test Cases" in h for h in headers)


class TestInferencePage:
    def test_navigate(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.run()
        at = _navigate_to_page(at, "Inference")
        assert not at.exception
        headers = [el.value for el in at.header]
        assert any("Inference" in h for h in headers)

    def test_agent_module_prefilled(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.run()
        at = _navigate_to_page(at, "Inference")
        module_input = at.text_input(key="eval_agent_module")
        assert module_input.value == "examples.weather_agent.agent"

    def test_not_reachable_without_eval_sets(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.run()
        assert at.button(key="nav_step_3").disabled


class TestEvaluationPage:
    def test_navigate(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.session_state["inference_results"] = _sample_inference_results()
        at.run()
        at = _navigate_to_page(at, "Evaluation")
        assert not at.exception
        headers = [el.value for el in at.header]
        assert any("Evaluation" in h for h in headers)

    def test_shows_metric_checkboxes(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.session_state["inference_results"] = _sample_inference_results()
        at.run()
        at = _navigate_to_page(at, "Evaluation")
        labels = [cb.label for cb in at.checkbox]
        assert "tool_trajectory_avg_score" in labels
        # safety_v1 should NOT be present (requires GCP)
        assert "safety_v1" not in labels

    def test_not_reachable_without_inference(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.run()
        assert at.button(key="nav_step_4").disabled


class TestFlowNavigation:
    def test_initial_step_is_zero(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert at.session_state["current_step"] == 0

    def test_first_two_steps_always_available(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert not at.button(key="nav_step_0").disabled
        assert not at.button(key="nav_step_1").disabled

    def test_step_unlocks_when_data_present(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert at.button(key="nav_step_2").disabled
        at.session_state["task_set"] = _sample_task_set()
        at = at.run()
        assert not at.button(key="nav_step_2").disabled

    def test_future_steps_disabled_without_data(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert at.button(key="nav_step_2").disabled
        assert at.button(key="nav_step_3").disabled
        assert at.button(key="nav_step_4").disabled

    def test_all_steps_unlock_with_full_data(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.session_state["inference_results"] = _sample_inference_results()
        at.session_state["eval_run_config"] = EvalRunConfig(
            agent_module="examples.weather_agent.agent",
            metrics=[MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8)])
        at.session_state["eval_results"] = _sample_eval_results()
        at.run()
        for i in range(len(PAGES)):
            assert not at.button(key=f"nav_step_{i}").disabled, f"Step {i} should be enabled"


class TestFullFlow:
    def test_navigate_all_pages_without_errors(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE).run()
        assert not at.exception
        for i in range(len(PAGES)):
            at = _navigate_to_step(at, i)
            assert not at.exception, f"Exception on page '{PAGES[i]}': {at.exception}"

    def test_full_flow_with_preloaded_data(self):
        _preload_metadata_env(_sample_metadata())
        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.session_state["inference_results"] = _sample_inference_results()
        at.session_state["eval_run_config"] = EvalRunConfig(
            agent_module="examples.weather_agent.agent",
            metrics=[MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8)])
        at.session_state["eval_results"] = _sample_eval_results()
        at.run()
        for i in range(len(PAGES)):
            at = _navigate_to_step(at, i)
            assert not at.exception, f"Exception on page '{PAGES[i]}': {at.exception}"
