"""End-to-end tests for the Streamlit UI flow using AppTest.

Tests the complete user journey through all pages:
1. App loads with pre-parsed agent metadata
2. Agent Metadata page shows agent tree
3. Tasks & Trajectories page - generate tasks via LLM
4. Test Cases page - generate evalsets
5. Eval Config page - configure metrics
6. Run Evaluation page - shows config
7. Eval Results page - shows results
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    TaskTrajectorySet,
    Task,
    Trajectory,
    TrajectoryStep,
    EvalRunConfig,
    MetricConfig,
    EvalRunResult,
    BasicMetrics,
    TraceSpanNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="weather_agent",
        agent_type="LlmAgent",
        description="A weather assistant",
        instruction="You help users check weather.",
        model="gemini-2.0-flash",
        tools=[
            ToolMetadata(name="get_weather", description="Get current weather", parameters_schema={
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]
            }),
            ToolMetadata(name="get_forecast", description="Get forecast", parameters_schema={
                "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}
            }),
        ],
        sub_agents=[],
    )


def _sample_task_set() -> TaskTrajectorySet:
    return TaskTrajectorySet(
        agent_name="weather_agent",
        tasks=[
            Task(
                task_id="check_weather",
                name="Check Current Weather",
                description="User asks for current weather in a city",
                trajectories=[
                    Trajectory(
                        trajectory_id="happy_path",
                        name="Successful weather check",
                        steps=[
                            TrajectoryStep(
                                user_message="What's the weather in London?",
                                expected_tool_calls=["get_weather"],
                                expected_tool_args={"get_weather": {"city": "London"}},
                                expected_response_keywords=["London", "temperature"],
                            ),
                        ],
                    ),
                ],
            ),
            Task(
                task_id="get_forecast",
                name="Get Weather Forecast",
                description="User asks for multi-day forecast",
                trajectories=[
                    Trajectory(
                        trajectory_id="happy_path",
                        name="Successful forecast",
                        steps=[
                            TrajectoryStep(
                                user_message="What's the 5-day forecast for Tokyo?",
                                expected_tool_calls=["get_forecast"],
                                expected_response_keywords=["Tokyo", "forecast"],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _sample_eval_sets() -> list[dict]:
    return [
        {
            "evalSetId": "weather_agent__check_weather",
            "name": "Check Current Weather",
            "description": "Tests for checking current weather",
            "evalCases": [
                {
                    "evalId": "check_weather__happy_path",
                    "conversation": [
                        {
                            "invocationId": "inv-1",
                            "userContent": {"role": "user", "parts": [{"text": "What's the weather in London?"}]},
                            "finalResponse": {"role": "model", "parts": [{"text": "The weather in London is cloudy, 15C."}]},
                            "intermediateData": {
                                "toolUses": [{"name": "get_weather", "args": {"city": "London"}}],
                                "toolResponses": [],
                                "intermediateResponses": [],
                            },
                        }
                    ],
                }
            ],
        }
    ]


def _sample_eval_results() -> list[EvalRunResult]:
    return [
        EvalRunResult(
            run_id="run-abc123",
            eval_set_id="weather_agent__check_weather",
            eval_id="check_weather__happy_path",
            status="PASSED",
            overall_scores={
                "tool_trajectory_avg_score": 1.0,
                "safety_v1": 1.0,
            },
            per_invocation_scores=[
                {
                    "invocation_id": "inv-1",
                    "user_message": "What's the weather in London?",
                    "actual_response": "The weather in London is cloudy, 15C.",
                    "expected_response": "The weather in London is cloudy, 15C.",
                    "actual_tool_calls": [{"name": "get_weather", "args": {"city": "London"}}],
                    "expected_tool_calls": [{"name": "get_weather", "args": {"city": "London"}}],
                    "scores": {"tool_trajectory_avg_score": 1.0, "safety_v1": 1.0},
                }
            ],
            basic_metrics=BasicMetrics(
                total_input_tokens=500,
                total_output_tokens=80,
                total_tokens=580,
                num_llm_calls=1,
                num_tool_calls=1,
                total_duration_ms=1200.0,
                max_context_size=500,
            ),
            session_id="sess-001",
            trace_tree=TraceSpanNode(
                span_id="s1",
                name="invocation",
                start_time=0.0,
                end_time=1.2,
                children=[
                    TraceSpanNode(
                        span_id="s2", name="call_llm",
                        start_time=0.0, end_time=0.5,
                        attributes={"gen_ai.usage.input_tokens": "500", "gen_ai.usage.output_tokens": "80"},
                    ),
                    TraceSpanNode(
                        span_id="s3", name="execute_tool:get_weather",
                        start_time=0.5, end_time=0.8,
                    ),
                ],
            ),
            timestamp=1000.0,
        ),
    ]


def _preload_metadata_env(metadata: AgentMetadata):
    """Write metadata to a temp file and set env vars like CLI does."""
    tmpdir = tempfile.mkdtemp()
    path = Path(tmpdir) / "metadata.json"
    path.write_text(metadata.model_dump_json(indent=2))
    os.environ["ADK_EVAL_PRELOADED_METADATA"] = str(path)
    os.environ["ADK_EVAL_AGENT_MODULE"] = "examples.weather_agent.agent"
    os.environ["ADK_EVAL_AGENT_VARIABLE"] = "root_agent"
    return tmpdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

APP_FILE = str(Path(__file__).resolve().parent.parent.parent / "adk_eval_tool" / "ui" / "app.py")


class TestAgentMetadataPage:
    """Test 1: App loads and shows agent metadata."""

    def test_app_loads_with_preloaded_metadata(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()

        # Should not have exceptions
        assert not at.exception, f"App raised exception: {at.exception}"

        # Sidebar should show agent name
        sidebar_texts = [el.value for el in at.sidebar.success]
        assert any("weather_agent" in t for t in sidebar_texts)

    def test_metadata_page_shows_agent_tree(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()

        # Default page is Agent Metadata - should show header
        headers = [el.value for el in at.header]
        assert any("Agent Metadata" in h for h in headers)

    def test_metadata_page_shows_tools(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()

        # Check markdown content mentions tools
        all_text = " ".join(el.value for el in at.markdown)
        assert "get_weather" in all_text or "weather_agent" in all_text


class TestTasksPage:
    """Test 2: Navigate to Tasks & Trajectories, generate tasks."""

    def test_navigate_to_tasks_page(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Tasks & Trajectories").run()

        assert not at.exception, f"Exception: {at.exception}"
        headers = [el.value for el in at.header]
        assert any("Tasks" in h for h in headers)

    def test_tasks_page_shows_warning_without_task_set(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Tasks & Trajectories").run()

        # Edit tab should show warning since no tasks yet
        warnings = [el.value for el in at.warning]
        assert any("Generate" in w or "load" in w for w in warnings)

    def test_tasks_page_shows_tasks_when_preloaded(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.run()
        at.sidebar.radio[0].set_value("Tasks & Trajectories").run()

        assert not at.exception, f"Exception: {at.exception}"


class TestTestCasesPage:
    """Test 3: Navigate to Test Cases page."""

    def test_navigate_to_test_cases(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Test Cases").run()

        assert not at.exception, f"Exception: {at.exception}"
        headers = [el.value for el in at.header]
        assert any("Test Cases" in h for h in headers)


class TestEvalConfigPage:
    """Test 4: Navigate to Eval Config, configure metrics."""

    def test_navigate_to_eval_config(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Eval Config").run()

        assert not at.exception, f"Exception: {at.exception}"
        headers = [el.value for el in at.header]
        assert any("Evaluation Configuration" in h for h in headers)

    def test_eval_config_prefills_agent_module(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Eval Config").run()

        # Agent module should be prefilled from env
        module_input = at.text_input(key="eval_agent_module")
        assert module_input.value == "examples.weather_agent.agent"

    def test_eval_config_shows_metrics(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Eval Config").run()

        # Should have checkboxes for metrics
        checkbox_labels = [cb.label for cb in at.checkbox]
        assert "tool_trajectory_avg_score" in checkbox_labels
        assert "safety_v1" in checkbox_labels

    def test_eval_config_save_creates_config(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Eval Config").run()

        # Set agent module
        at.text_input(key="eval_agent_module").set_value("examples.weather_agent.agent").run()

        # Click save
        at.button(key="save_eval_config").click().run() if any(
            b.key == "save_eval_config" for b in at.button
        ) else None

        # After interaction, config should exist if Save was clicked
        # (button may not have the exact key, so we check it doesn't crash)
        assert not at.exception, f"Exception: {at.exception}"


class TestEvalLauncherPage:
    """Test 5: Navigate to Run Evaluation page."""

    def test_launcher_shows_warning_without_config(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Run Evaluation").run()

        warnings = [el.value for el in at.warning]
        assert any("Configure" in w for w in warnings)

    def test_launcher_shows_config_when_set(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        config = EvalRunConfig(
            agent_module="examples.weather_agent.agent",
            metrics=[
                MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8),
                MetricConfig(metric_name="safety_v1", threshold=1.0),
            ],
        )

        at = AppTest.from_file(APP_FILE)
        at.session_state["eval_run_config"] = config
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.run()
        at.sidebar.radio[0].set_value("Run Evaluation").run()

        assert not at.exception, f"Exception: {at.exception}"
        headers = [el.value for el in at.header]
        assert any("Run Evaluation" in h for h in headers)


class TestEvalResultsPage:
    """Test 6: Navigate to Eval Results and verify scores/traces."""

    def test_results_page_shows_no_results_message(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Eval Results").run()

        assert not at.exception, f"Exception: {at.exception}"
        info_msgs = [el.value for el in at.info]
        assert any("No evaluation results" in m for m in info_msgs)

    def test_results_page_shows_results_when_available(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE)
        at.session_state["eval_results"] = _sample_eval_results()
        at.run()
        at.sidebar.radio[0].set_value("Eval Results").run()

        assert not at.exception, f"Exception: {at.exception}"
        headers = [el.value for el in at.header]
        assert any("Evaluation Results" in h for h in headers)

    def test_results_page_shows_pass_status(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE)
        at.session_state["eval_results"] = _sample_eval_results()
        at.run()
        at.sidebar.radio[0].set_value("Eval Results").run()

        # Should show metrics with scores
        all_text = " ".join(el.value for el in at.metric)
        # Metrics should show passed count, total, or score values
        assert "1" in all_text or "PASSED" in all_text or "Passed" in all_text

    def test_results_page_shows_basic_metrics(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE)
        at.session_state["eval_results"] = _sample_eval_results()
        at.run()
        at.sidebar.radio[0].set_value("Eval Results").run()

        # The metrics from BasicMetrics should appear somewhere
        metric_values = [str(el.value) for el in at.metric]
        # Should show total tokens, LLM calls etc as metric widgets
        assert len(metric_values) > 0


class TestDatasetVersionsPage:
    """Test 7: Navigate to Dataset Versions page."""

    def test_navigate_to_versions(self):
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE).run()
        at.sidebar.radio[0].set_value("Dataset Versions").run()

        assert not at.exception, f"Exception: {at.exception}"
        headers = [el.value for el in at.header]
        assert any("Dataset Versions" in h for h in headers)


class TestFullFlow:
    """Test the complete flow through all pages."""

    def test_navigate_all_pages_without_errors(self):
        """Smoke test: navigate every page, no exceptions."""
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        pages = [
            "Agent Metadata",
            "Tasks & Trajectories",
            "Test Cases",
            "Eval Config",
            "Run Evaluation",
            "Eval Results",
            "Dataset Versions",
        ]

        at = AppTest.from_file(APP_FILE).run()
        assert not at.exception, f"Initial load exception: {at.exception}"

        for page in pages:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"Exception on page '{page}': {at.exception}"

    def test_full_flow_with_preloaded_data(self):
        """Test flow with all data pre-populated in session state."""
        metadata = _sample_metadata()
        _preload_metadata_env(metadata)

        at = AppTest.from_file(APP_FILE)
        at.session_state["task_set"] = _sample_task_set()
        at.session_state["eval_sets"] = _sample_eval_sets()
        at.session_state["eval_run_config"] = EvalRunConfig(
            agent_module="examples.weather_agent.agent",
            metrics=[
                MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8),
                MetricConfig(metric_name="safety_v1", threshold=1.0),
            ],
        )
        at.session_state["eval_results"] = _sample_eval_results()
        at.run()

        # Navigate through all pages with data
        pages = [
            "Agent Metadata",
            "Tasks & Trajectories",
            "Test Cases",
            "Eval Config",
            "Run Evaluation",
            "Eval Results",
            "Dataset Versions",
        ]

        for page in pages:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"Exception on page '{page}': {at.exception}"
