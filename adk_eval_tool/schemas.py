"""Pydantic models for agent metadata, intents, scenarios, and test config."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# --- Agent Metadata ---


class ToolMetadata(BaseModel):
    """Metadata for a single tool (function tool or MCP tool)."""

    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    source: str = "function"  # "function" | "mcp" | "builtin"
    mcp_server_name: Optional[str] = None


class AgentMetadata(BaseModel):
    """Recursive metadata tree for an ADK agent and its sub-agents."""

    name: str
    agent_type: str  # "LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent", "Custom"
    description: str = ""
    instruction: str = ""
    model: str = ""
    tools: list[ToolMetadata] = Field(default_factory=list)
    sub_agents: list[AgentMetadata] = Field(default_factory=list)
    output_key: Optional[str] = None
    disallow_transfer_to_parent: bool = False
    disallow_transfer_to_peers: bool = False


# --- Intents & Scenarios ---


class Scenario(BaseModel):
    """A test scenario for a task.

    Describes a combination of input type (intent variation) and expected
    output type in plain text.  Scenarios do not prescribe specific
    conversation turns — those are generated at the test-case stage.
    """

    scenario_id: str
    name: str
    description: str = ""
    eval_type: str = "scenario"  # "scenario" | "conversation_scenario"
    conversation_scenario: Optional[dict[str, str]] = None
    session_state: Optional[dict[str, Any]] = None


class Task(BaseModel):
    """A user task with associated scenarios."""

    task_id: str
    name: str
    description: str
    scenarios: list[Scenario] = Field(default_factory=list)


class TaskScenarioSet(BaseModel):
    """Complete set of tasks and scenarios for an agent."""

    agent_name: str
    tasks: list[Task] = Field(default_factory=list)
    generation_context: str = ""
    version: str = "1"


# --- Test Case Config ---


class MultiTurnConfig(BaseModel):
    """Configuration for multi-turn test case generation."""

    enabled: bool = True
    min_turns: int = 2
    max_turns: int = 5
    include_clarification: bool = True
    include_correction: bool = True
    include_follow_up: bool = True


class ScenarioWeight(BaseModel):
    """A scenario or failure type with its percentage weight."""

    name: str
    weight: float = 0.0  # percentage of total test cases (0-100)


class TestGenConfig(BaseModel):
    """Configuration for test case generation (LLM-based)."""

    total_test_cases_per_task: int = 10
    multi_turn: MultiTurnConfig = Field(default_factory=MultiTurnConfig)
    scenario_weights: list[ScenarioWeight] = Field(default_factory=lambda: [
        ScenarioWeight(name="happy_path", weight=30),
        ScenarioWeight(name="failure_path", weight=30),
        ScenarioWeight(name="edge_case", weight=20),
        ScenarioWeight(name="multi_turn", weight=20),
    ])
    failure_weights: list[ScenarioWeight] = Field(default_factory=lambda: [
        ScenarioWeight(name="missing_required_input", weight=25),
        ScenarioWeight(name="invalid_input_format", weight=25),
        ScenarioWeight(name="tool_error", weight=25),
        ScenarioWeight(name="ambiguous_request", weight=25),
    ])
    num_simulations_per_task: int = 3
    judge_model: str = "gemini-2.5-flash"
    tool_trajectory_match_type: str = "IN_ORDER"


class TestCaseConfig(BaseModel):
    """Configuration for test case generation and evaluation."""

    eval_metrics: dict[str, float] = Field(
        default_factory=lambda: {
            "tool_trajectory_avg_score": 0.8,
        }
    )
    judge_model: str = "gemini-2.5-flash"
    num_runs: int = 2
    tool_trajectory_match_type: str = "IN_ORDER"


# --- Eval Run & Results ---


class MetricConfig(BaseModel):
    """Configuration for a single evaluation metric."""

    metric_name: str
    threshold: float = 0.8
    match_type: Optional[str] = None
    judge_model: Optional[str] = None
    judge_num_samples: int = 5
    evaluate_intermediate: bool = False
    rubric: Optional[str] = None


class UserSimulatorConfig(BaseModel):
    """Configuration for ADK user simulator (conversation_scenario tests)."""

    model: str = "gemini-2.5-flash"
    max_allowed_invocations: int = 20
    custom_instructions: Optional[str] = None


class CustomMetricDef(BaseModel):
    """Definition of a custom evaluation metric."""

    name: str
    code_path: str  # e.g., "my_module.my_metric_function"
    description: str = ""


class EvalRunConfig(BaseModel):
    """Configuration for running an evaluation."""

    agent_module: str
    agent_name: Optional[str] = None
    metrics: list[MetricConfig] = Field(default_factory=list)
    judge_model: str = "gemini-2.5-flash"
    num_runs: int = 2
    trace_db_path: str = "eval_traces.db"
    user_simulator: Optional[UserSimulatorConfig] = None
    custom_metrics: list[CustomMetricDef] = Field(default_factory=list)


class TraceSpanNode(BaseModel):
    """A node in a trace tree, representing one OTel span."""

    span_id: str
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    children: list[TraceSpanNode] = Field(default_factory=list)


class BasicMetrics(BaseModel):
    """Standard non-LLM metrics collected during evaluation runs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    total_duration_ms: float = 0.0
    avg_response_length: float = 0.0
    max_context_size: int = 0


class InvocationScore(BaseModel):
    """Scores for a single invocation turn."""

    invocation_id: str
    user_message: str = ""
    actual_response: str = ""
    expected_response: str = ""
    actual_tool_calls: list[str] = Field(default_factory=list)
    expected_tool_calls: list[str] = Field(default_factory=list)
    actual_tool_args: list[dict[str, Any]] = Field(default_factory=list)
    expected_tool_args: list[dict[str, Any]] = Field(default_factory=list)
    scores: dict[str, Optional[float]] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


class InferenceRunResult(BaseModel):
    """Result of inference for a single eval case (no scoring).

    Captures what the agent actually did without judging correctness.
    The raw inference_result_json is preserved so that offline evaluation
    can be run later without re-running the agent.
    """

    run_id: str
    eval_set_id: str
    eval_id: str
    session_id: str = ""
    inference_result_json: dict[str, Any] = Field(default_factory=dict)
    eval_set_json: dict[str, Any] = Field(default_factory=dict)
    actual_invocations: list[dict[str, Any]] = Field(default_factory=list)
    basic_metrics: Optional[BasicMetrics] = None
    trace_tree: Optional[TraceSpanNode] = None
    timestamp: float = 0.0


class EvalRunResult(BaseModel):
    """Result of a single eval case run (one scenario, one run)."""

    run_id: str
    eval_set_id: str
    eval_id: str
    status: str  # "PASSED" | "FAILED" | "NOT_EVALUATED"
    overall_scores: dict[str, Optional[float]] = Field(default_factory=dict)
    per_invocation_scores: list[dict[str, Any]] = Field(default_factory=list)
    basic_metrics: Optional[BasicMetrics] = None
    session_id: str = ""
    trace_tree: Optional[TraceSpanNode] = None
    timestamp: float = 0.0
