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


class ScenarioStep(BaseModel):
    """A single conversation turn in a scenario.

    Models a full ADK Invocation: user message -> tool trajectory -> response.
    For trajectory-based evals, expected_tool_calls and expected_tool_args
    define the expected tool use sequence. tool_responses provides reference
    data for hallucination metrics. intermediate_responses captures expected
    agent reasoning between tool calls.
    """

    user_message: str
    expected_tool_calls: list[str] = Field(default_factory=list)
    expected_tool_args: Optional[dict[str, dict[str, Any]]] = None
    tool_responses: Optional[dict[str, Any]] = None
    expected_response: str = ""
    expected_response_keywords: list[str] = Field(default_factory=list)
    intermediate_responses: list[dict[str, Any]] = Field(default_factory=list)
    rubric: Optional[str] = None
    notes: str = ""


class Scenario(BaseModel):
    """A multi-turn test scenario for a specific intent.

    Supports two modes matching ADK's EvalCase:
    - Static trajectory: steps[] defines fixed conversation turns (maps to ADK 'conversation')
    - Dynamic simulation: conversation_scenario defines a starting prompt + plan
      (maps to ADK 'conversation_scenario' with user simulator)
    """

    scenario_id: str
    name: str
    description: str = ""
    eval_type: str = "trajectory"  # "trajectory" | "conversation_scenario"
    steps: list[ScenarioStep] = Field(default_factory=list)
    conversation_scenario: Optional[dict[str, str]] = None
    session_state: Optional[dict[str, Any]] = None
    tags: list[str] = Field(default_factory=list)


class Intent(BaseModel):
    """A user intent with associated test scenarios."""

    intent_id: str
    name: str
    description: str
    category: str = ""
    scenarios: list[Scenario] = Field(default_factory=list)


class IntentScenarioSet(BaseModel):
    """Complete set of intents and scenarios for an agent."""

    agent_name: str
    intents: list[Intent] = Field(default_factory=list)
    generation_context: str = ""
    version: str = "1"


# --- Test Case Config ---


class TestCaseConfig(BaseModel):
    """Configuration for test case generation and evaluation."""

    eval_metrics: dict[str, float] = Field(
        default_factory=lambda: {
            "tool_trajectory_avg_score": 0.8,
            "safety_v1": 1.0,
        }
    )
    judge_model: str = "gemini-2.0-flash"
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


class EvalRunConfig(BaseModel):
    """Configuration for running an evaluation."""

    agent_module: str
    agent_name: Optional[str] = None
    metrics: list[MetricConfig] = Field(default_factory=list)
    judge_model: str = "gemini-2.5-flash"
    num_runs: int = 2
    trace_db_path: str = "eval_traces.db"


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
