# ADK Evaluation & Testing Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python toolkit + Streamlit UI that introspects Google ADK agents, generates evaluation intents/scenarios/test cases via ADK agents, and manages golden datasets.

**Architecture:** Six modules — (1) `agent_parser` introspects live ADK Agent objects into a metadata tree, (2) `intent_generator` is an ADK agent that produces intents + scenarios from metadata, (3) `testcase_generator` is an ADK agent that produces ADK-compatible `.evalset.json` files and self-validates them, (4) `eval_runner` runs evaluations with trace collection via OpenTelemetry/SqliteSpanExporter and captures structured results via LocalEvalService, (5) shared `schemas` define all intermediate data structures, (6) `ui` is a Streamlit app tying everything together with dedicated pages for eval config, test execution, and trace/result exploration. Each module is independently usable as a Python API.

**Tech Stack:** Python 3.10+, google-adk, Pydantic v2, Streamlit, google-genai types, OpenTelemetry SDK

---

## File Structure

```
adk_eval_tool/
├── __init__.py                        # Package exports
├── schemas.py                         # Pydantic models for metadata, intents, scenarios, config
├── agent_parser/
│   ├── __init__.py
│   ├── parser.py                      # parse_agent() — main entry point
│   └── mcp_resolver.py               # MCP tool discovery helper
├── intent_generator/
│   ├── __init__.py
│   ├── agent.py                       # ADK intent/scenario generator agent definition
│   ├── prompts.py                     # System instructions and prompt templates
│   └── tools.py                       # Function tools for the generator agent
├── testcase_generator/
│   ├── __init__.py
│   ├── agent.py                       # ADK test case generator agent definition
│   ├── prompts.py                     # System instructions and prompt templates
│   └── tools.py                       # Function tools (validation, file writing)
├── eval_runner/
│   ├── __init__.py
│   ├── runner.py                      # Run evaluations with trace + result capture
│   ├── trace_collector.py             # OTel setup, SqliteSpanExporter wrapper, span→tree
│   └── result_store.py                # Persist & query EvalCaseResults + EvalSetResults
├── ui/
│   ├── __init__.py
│   ├── app.py                         # Streamlit main app + navigation
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── metadata_viewer.py         # Page: parse agent, view/edit metadata
│   │   ├── intent_manager.py          # Page: generate/edit intents & scenarios
│   │   ├── testcase_manager.py        # Page: generate/edit test cases
│   │   ├── eval_config.py             # Page: configure metrics, thresholds, judge model
│   │   ├── eval_launcher.py           # Page: launch eval runs, monitor progress
│   │   ├── eval_results.py            # Page: explore results, trace trees, scores
│   │   └── dataset_versions.py        # Page: version management
│   └── components/
│       ├── __init__.py
│       ├── json_editor.py             # Reusable JSON tree editor widget
│       ├── run_status.py              # Generation progress/status display
│       └── trace_tree.py              # Trace tree visualization component
tests/
├── __init__.py
├── conftest.py                        # Shared fixtures (sample agents, metadata)
├── test_schemas.py                    # Schema validation tests
├── test_parser.py                     # Agent parser tests
├── test_mcp_resolver.py              # MCP resolver tests
├── test_intent_generator.py          # Intent generator tests
├── test_testcase_generator.py        # Test case generator tests
├── test_eval_runner.py               # Eval runner + trace collection tests
├── test_result_store.py              # Result store tests
└── test_ui_components.py             # UI component unit tests
pyproject.toml                         # Project config, dependencies
```

---

## Task 1: Project Setup & Schemas

**Files:**
- Create: `pyproject.toml`
- Create: `adk_eval_tool/__init__.py`
- Create: `adk_eval_tool/schemas.py`
- Create: `tests/__init__.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "adk-eval-tool"
version = "0.1.0"
description = "Agentic evaluation and testing tool for Google ADK agents"
requires-python = ">=3.10"
dependencies = [
    "google-adk[eval]>=1.0.0",
    "google-genai>=1.0.0",
    "pydantic>=2.0",
    "streamlit>=1.30.0",
    "opentelemetry-sdk>=1.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write schema tests**

```python
# tests/test_schemas.py
import json
from adk_eval_tool.schemas import (
    ToolMetadata,
    AgentMetadata,
    Intent,
    Scenario,
    ScenarioStep,
    IntentScenarioSet,
    TestCaseConfig,
    EvalRunConfig,
    MetricConfig,
    EvalRunResult,
    TraceSpanNode,
    BasicMetrics,
)


def test_tool_metadata_basic():
    tool = ToolMetadata(
        name="search",
        description="Search the web",
        parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert tool.name == "search"
    assert tool.source == "function"


def test_tool_metadata_mcp():
    tool = ToolMetadata(
        name="read_file",
        description="Read a file",
        parameters_schema={},
        source="mcp",
        mcp_server_name="filesystem",
    )
    assert tool.source == "mcp"
    assert tool.mcp_server_name == "filesystem"


def test_agent_metadata_with_sub_agents():
    child = AgentMetadata(
        name="researcher",
        agent_type="LlmAgent",
        description="Research assistant",
        instruction="You research topics.",
        model="gemini-2.0-flash",
        tools=[ToolMetadata(name="search", description="Search", parameters_schema={})],
        sub_agents=[],
    )
    root = AgentMetadata(
        name="coordinator",
        agent_type="LlmAgent",
        description="Main coordinator",
        instruction="You coordinate tasks.",
        model="gemini-2.0-flash",
        tools=[],
        sub_agents=[child],
    )
    assert len(root.sub_agents) == 1
    assert root.sub_agents[0].name == "researcher"


def test_agent_metadata_serialization_roundtrip():
    meta = AgentMetadata(
        name="test_agent",
        agent_type="LlmAgent",
        description="Test",
        instruction="Do things.",
        model="gemini-2.0-flash",
        tools=[],
        sub_agents=[],
    )
    data = json.loads(meta.model_dump_json())
    restored = AgentMetadata.model_validate(data)
    assert restored.name == meta.name


def test_scenario_step():
    step = ScenarioStep(
        user_message="Book a flight to London",
        expected_tool_calls=["search_flights", "book_flight"],
        expected_tool_args={"search_flights": {"destination": "London"}},
        tool_responses={"search_flights": {"flights": [{"id": "FL-1", "price": 200}]}},
        expected_response="I found a flight to London for $200.",
        expected_response_keywords=["booked", "London"],
        rubric="Agent should confirm the booking details before proceeding.",
        notes="Happy path booking",
    )
    assert len(step.expected_tool_calls) == 2
    assert step.tool_responses is not None
    assert step.rubric is not None


def test_scenario_conversation_scenario_type():
    scenario = Scenario(
        scenario_id="dynamic_refund",
        name="Dynamic refund flow",
        eval_type="conversation_scenario",
        conversation_scenario={
            "starting_prompt": "I want a refund for order ORD-123",
            "conversation_plan": "Provide order ID when asked. Confirm refund. Signal completion.",
        },
        session_state={"user_tier": "premium"},
        tags=["dynamic", "multi_turn"],
    )
    assert scenario.eval_type == "conversation_scenario"
    assert scenario.conversation_scenario is not None
    assert scenario.session_state["user_tier"] == "premium"


def test_intent_with_scenarios():
    scenario = Scenario(
        scenario_id="book_flight_happy",
        name="Successful flight booking",
        description="User books a flight successfully",
        steps=[
            ScenarioStep(
                user_message="Book a flight to London tomorrow",
                expected_tool_calls=["search_flights"],
            )
        ],
        tags=["happy_path"],
    )
    intent = Intent(
        intent_id="book_flight",
        name="Book Flight",
        description="User wants to book a flight",
        category="booking",
        scenarios=[scenario],
    )
    assert intent.intent_id == "book_flight"
    assert len(intent.scenarios) == 1


def test_intent_scenario_set():
    intent_set = IntentScenarioSet(
        agent_name="travel_agent",
        intents=[
            Intent(
                intent_id="book_flight",
                name="Book Flight",
                description="Book a flight",
                category="booking",
                scenarios=[],
            )
        ],
        generation_context="Testing travel agent",
    )
    assert intent_set.agent_name == "travel_agent"


def test_testcase_config():
    config = TestCaseConfig(
        eval_metrics={"tool_trajectory_avg_score": 0.8, "safety_v1": 1.0},
        judge_model="gemini-2.0-flash",
    )
    assert config.eval_metrics["safety_v1"] == 1.0


def test_metric_config():
    mc = MetricConfig(
        metric_name="tool_trajectory_avg_score",
        threshold=0.9,
        match_type="IN_ORDER",
    )
    assert mc.metric_name == "tool_trajectory_avg_score"
    assert mc.match_type == "IN_ORDER"


def test_eval_run_config():
    erc = EvalRunConfig(
        agent_module="my_agent.agent",
        metrics=[
            MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8),
            MetricConfig(metric_name="safety_v1", threshold=1.0),
        ],
        judge_model="gemini-2.5-flash",
        num_runs=3,
    )
    assert erc.agent_module == "my_agent.agent"
    assert len(erc.metrics) == 2


def test_trace_span_node():
    child = TraceSpanNode(
        span_id="span_2",
        name="execute_tool:search",
        start_time=1000.1,
        end_time=1000.5,
        attributes={"gcp.vertex.agent.event_id": "evt-2"},
    )
    parent = TraceSpanNode(
        span_id="span_1",
        name="call_llm",
        start_time=1000.0,
        end_time=1001.0,
        attributes={},
        children=[child],
    )
    assert len(parent.children) == 1
    assert parent.children[0].name == "execute_tool:search"


def test_eval_run_result():
    result = EvalRunResult(
        run_id="run-123",
        eval_set_id="agent__intent",
        eval_id="intent__scenario",
        status="PASSED",
        overall_scores={"tool_trajectory_avg_score": 0.95, "safety_v1": 1.0},
        per_invocation_scores=[
            {"invocation_id": "inv-1", "scores": {"tool_trajectory_avg_score": 0.95}},
        ],
        basic_metrics=BasicMetrics(
            total_input_tokens=1500,
            total_output_tokens=300,
            total_tokens=1800,
            num_llm_calls=2,
            num_tool_calls=3,
            total_duration_ms=4500.0,
            avg_response_length=120.5,
            max_context_size=1200,
        ),
    )
    assert result.status == "PASSED"
    assert result.overall_scores["safety_v1"] == 1.0
    assert result.basic_metrics.total_tokens == 1800
    assert result.basic_metrics.num_tool_calls == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/vladimirz/Work/adk-eval-ext && python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adk_eval_tool'`

- [ ] **Step 4: Implement schemas**

```python
# adk_eval_tool/__init__.py
"""ADK Evaluation & Testing Tool."""
```

```python
# adk_eval_tool/schemas.py
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

    Models a full ADK Invocation: user message → tool trajectory → response.
    For trajectory-based evals, expected_tool_calls and expected_tool_args
    define the expected tool use sequence. tool_responses provides reference
    data for hallucination metrics. intermediate_responses captures expected
    agent reasoning between tool calls.
    """

    user_message: str
    expected_tool_calls: list[str] = Field(default_factory=list)
    expected_tool_args: Optional[dict[str, dict[str, Any]]] = None
    tool_responses: Optional[dict[str, Any]] = None  # Reference tool output for hallucination eval
    expected_response: str = ""  # Expected final response text
    expected_response_keywords: list[str] = Field(default_factory=list)
    intermediate_responses: list[dict[str, Any]] = Field(default_factory=list)  # Agent reasoning between tool calls
    rubric: Optional[str] = None  # Per-invocation quality rubric
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
    steps: list[ScenarioStep] = Field(default_factory=list)  # For trajectory-based
    conversation_scenario: Optional[dict[str, str]] = None  # For dynamic: {"starting_prompt": ..., "conversation_plan": ...}
    session_state: Optional[dict[str, Any]] = None  # Initial session state
    tags: list[str] = Field(default_factory=list)  # e.g. "happy_path", "edge_case", "error"


class Intent(BaseModel):
    """A user intent with associated test scenarios."""

    intent_id: str
    name: str
    description: str
    category: str = ""  # e.g. "booking", "inquiry", "error_handling"
    scenarios: list[Scenario] = Field(default_factory=list)


class IntentScenarioSet(BaseModel):
    """Complete set of intents and scenarios for an agent."""

    agent_name: str
    intents: list[Intent] = Field(default_factory=list)
    generation_context: str = ""  # user-provided constraints/context
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
    tool_trajectory_match_type: str = "IN_ORDER"  # EXACT | IN_ORDER | ANY_ORDER


# --- Eval Run & Results ---


class MetricConfig(BaseModel):
    """Configuration for a single evaluation metric."""

    metric_name: str  # One of ADK PrebuiltMetrics values
    threshold: float = 0.8
    match_type: Optional[str] = None  # For tool_trajectory: EXACT | IN_ORDER | ANY_ORDER
    judge_model: Optional[str] = None  # Override judge model for this metric
    judge_num_samples: int = 5
    evaluate_intermediate: bool = False  # For hallucinations_v1
    rubric: Optional[str] = None  # For rubric-based metrics


class EvalRunConfig(BaseModel):
    """Configuration for running an evaluation."""

    agent_module: str  # Python module path (e.g., "my_agent.agent")
    agent_name: Optional[str] = None  # Specific agent name (None = root_agent)
    metrics: list[MetricConfig] = Field(default_factory=list)
    judge_model: str = "gemini-2.5-flash"
    num_runs: int = 2
    trace_db_path: str = "eval_traces.db"  # SQLite path for span storage


class TraceSpanNode(BaseModel):
    """A node in a trace tree, representing one OTel span."""

    span_id: str
    name: str  # e.g., "invocation", "call_llm", "execute_tool:search"
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
    avg_response_length: float = 0.0  # Average chars in final responses
    max_context_size: int = 0  # Largest input token count in a single LLM call


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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/vladimirz/Work/adk-eval-ext && python -m pytest tests/test_schemas.py -v`
Expected: All 14 tests PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml adk_eval_tool/__init__.py adk_eval_tool/schemas.py tests/__init__.py tests/test_schemas.py
git commit -m "feat: add project setup and Pydantic schemas for metadata, intents, scenarios"
```

---

## Task 2: Agent Parser — Core Parsing

**Files:**
- Create: `adk_eval_tool/agent_parser/__init__.py`
- Create: `adk_eval_tool/agent_parser/parser.py`
- Create: `tests/conftest.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write conftest with sample agents**

```python
# tests/conftest.py
"""Shared test fixtures."""

import pytest
from google.adk.agents import Agent, SequentialAgent


def _dummy_tool(query: str) -> str:
    """Search for information.

    Args:
        query: The search query string.

    Returns:
        Search results as text.
    """
    return f"Results for {query}"


def _another_tool(text: str, max_length: int = 100) -> str:
    """Summarize text to a shorter version.

    Args:
        text: The text to summarize.
        max_length: Maximum length of the summary.

    Returns:
        Summarized text.
    """
    return text[:max_length]


@pytest.fixture
def simple_agent():
    return Agent(
        name="simple_agent",
        model="gemini-2.0-flash",
        description="A simple test agent",
        instruction="You are a helpful assistant. Answer questions clearly.",
        tools=[_dummy_tool],
    )


@pytest.fixture
def agent_with_sub_agents():
    researcher = Agent(
        name="researcher",
        model="gemini-2.0-flash",
        description="Researches topics",
        instruction="You research topics using search.",
        tools=[_dummy_tool],
    )
    writer = Agent(
        name="writer",
        model="gemini-2.0-flash",
        description="Writes summaries",
        instruction="You write clear summaries.",
        tools=[_another_tool],
    )
    coordinator = Agent(
        name="coordinator",
        model="gemini-2.0-flash",
        description="Coordinates research and writing",
        instruction="You coordinate the researcher and writer.",
        sub_agents=[researcher, writer],
    )
    return coordinator


@pytest.fixture
def sequential_agent():
    step1 = Agent(
        name="step1",
        model="gemini-2.0-flash",
        instruction="First step.",
        output_key="step1_output",
    )
    step2 = Agent(
        name="step2",
        model="gemini-2.0-flash",
        instruction="Second step using {step1_output}.",
    )
    return SequentialAgent(
        name="pipeline",
        description="A sequential pipeline",
        sub_agents=[step1, step2],
    )
```

- [ ] **Step 2: Write parser tests**

```python
# tests/test_parser.py
"""Tests for agent parser."""

from adk_eval_tool.agent_parser.parser import parse_agent
from adk_eval_tool.schemas import AgentMetadata


def test_parse_simple_agent(simple_agent):
    metadata = parse_agent(simple_agent)
    assert isinstance(metadata, AgentMetadata)
    assert metadata.name == "simple_agent"
    assert metadata.agent_type == "LlmAgent"
    assert metadata.description == "A simple test agent"
    assert "helpful assistant" in metadata.instruction
    assert metadata.model == "gemini-2.0-flash"
    assert len(metadata.tools) == 1
    assert metadata.tools[0].name == "_dummy_tool"
    assert metadata.tools[0].source == "function"
    assert "query" in str(metadata.tools[0].parameters_schema)


def test_parse_agent_with_sub_agents(agent_with_sub_agents):
    metadata = parse_agent(agent_with_sub_agents)
    assert metadata.name == "coordinator"
    assert len(metadata.sub_agents) == 2
    researcher = metadata.sub_agents[0]
    assert researcher.name == "researcher"
    assert len(researcher.tools) == 1
    assert researcher.tools[0].name == "_dummy_tool"
    writer = metadata.sub_agents[1]
    assert writer.name == "writer"
    assert len(writer.tools) == 1
    assert writer.tools[0].name == "_another_tool"


def test_parse_sequential_agent(sequential_agent):
    metadata = parse_agent(sequential_agent)
    assert metadata.name == "pipeline"
    assert metadata.agent_type == "SequentialAgent"
    assert len(metadata.sub_agents) == 2
    assert metadata.sub_agents[0].output_key == "step1_output"


def test_parse_agent_tool_schema(simple_agent):
    metadata = parse_agent(simple_agent)
    tool = metadata.tools[0]
    schema = tool.parameters_schema
    assert "properties" in schema
    assert "query" in schema["properties"]


def test_parse_returns_dict(simple_agent):
    metadata = parse_agent(simple_agent)
    as_dict = metadata.model_dump()
    assert isinstance(as_dict, dict)
    assert as_dict["name"] == "simple_agent"


def test_parse_save_to_file(simple_agent, tmp_path):
    output_path = tmp_path / "metadata.json"
    metadata = parse_agent(simple_agent, save_path=str(output_path))
    assert output_path.exists()
    import json
    loaded = json.loads(output_path.read_text())
    assert loaded["name"] == "simple_agent"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adk_eval_tool.agent_parser'`

- [ ] **Step 4: Implement agent parser**

```python
# adk_eval_tool/agent_parser/__init__.py
"""Agent parser — introspect ADK agents into metadata trees."""

from adk_eval_tool.agent_parser.parser import parse_agent

__all__ = ["parse_agent"]
```

```python
# adk_eval_tool/agent_parser/parser.py
"""Parse a live ADK Agent object into an AgentMetadata tree."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional, Union

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

from adk_eval_tool.schemas import AgentMetadata, ToolMetadata


_AGENT_TYPE_MAP = {
    LlmAgent: "LlmAgent",
    SequentialAgent: "SequentialAgent",
    ParallelAgent: "ParallelAgent",
    LoopAgent: "LoopAgent",
}


def _get_agent_type(agent: BaseAgent) -> str:
    for cls, name in _AGENT_TYPE_MAP.items():
        if isinstance(agent, cls):
            return name
    return type(agent).__name__


def _extract_function_schema(func: callable) -> dict[str, Any]:
    """Extract JSON schema from a Python function's signature."""
    sig = inspect.signature(func)
    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "ctx", "tool_context", "context"):
            continue
        # Check if the annotation looks like a ToolContext
        ann = param.annotation
        if ann != inspect.Parameter.empty:
            ann_name = getattr(ann, "__name__", str(ann))
            if "Context" in ann_name:
                continue

        prop: dict[str, Any] = {}
        if ann != inspect.Parameter.empty:
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
            prop["type"] = type_map.get(ann, "string")
        else:
            prop["type"] = "string"

        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default

        properties[param_name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parse_tool(tool: Union[callable, BaseTool]) -> ToolMetadata:
    """Parse a single tool into ToolMetadata."""
    if isinstance(tool, BaseTool):
        return ToolMetadata(
            name=tool.name,
            description=tool.description or "",
            parameters_schema=_extract_declaration_schema(tool),
            source="function" if isinstance(tool, FunctionTool) else "builtin",
        )
    elif callable(tool):
        return ToolMetadata(
            name=tool.__name__,
            description=inspect.cleandoc(tool.__doc__ or ""),
            parameters_schema=_extract_function_schema(tool),
            source="function",
        )
    else:
        return ToolMetadata(name=str(tool), description="", parameters_schema={})


def _extract_declaration_schema(tool: BaseTool) -> dict[str, Any]:
    """Try to get schema from a BaseTool's declaration."""
    try:
        decl = tool._get_declaration()
        if decl and decl.parameters:
            # Convert Schema object to dict
            schema = {"type": "object", "properties": {}}
            if hasattr(decl.parameters, "properties") and decl.parameters.properties:
                for prop_name, prop_schema in decl.parameters.properties.items():
                    schema["properties"][prop_name] = {
                        "type": getattr(prop_schema, "type", "string"),
                    }
            if hasattr(decl.parameters, "required") and decl.parameters.required:
                schema["required"] = list(decl.parameters.required)
            return schema
    except Exception:
        pass
    return {}


async def _resolve_toolset_tools(toolset: BaseToolset) -> list[ToolMetadata]:
    """Resolve tools from a toolset (e.g., MCPToolset)."""
    try:
        tools = await toolset.get_tools()
        return [_parse_tool(t) for t in tools]
    except Exception:
        return [ToolMetadata(
            name=f"unresolved_toolset_{type(toolset).__name__}",
            description=f"Could not resolve tools from {type(toolset).__name__}",
            parameters_schema={},
            source="mcp",
        )]


def _parse_tools_sync(tools_list: list) -> list[ToolMetadata]:
    """Parse tools synchronously. Toolsets that need async are deferred."""
    result = []
    for tool in tools_list:
        if isinstance(tool, BaseToolset):
            # Mark for async resolution
            result.append(ToolMetadata(
                name=f"toolset:{type(tool).__name__}",
                description=f"Toolset requiring async resolution",
                parameters_schema={},
                source="mcp",
            ))
        else:
            result.append(_parse_tool(tool))
    return result


def _parse_agent_recursive(agent: BaseAgent) -> AgentMetadata:
    """Recursively parse an agent into AgentMetadata."""
    instruction = ""
    model = ""
    tools: list[ToolMetadata] = []
    output_key = None
    disallow_transfer_to_parent = False
    disallow_transfer_to_peers = False

    if isinstance(agent, LlmAgent):
        # instruction can be a string or callable
        raw_instruction = agent.instruction
        if isinstance(raw_instruction, str):
            instruction = raw_instruction
        elif callable(raw_instruction):
            instruction = f"<dynamic: {raw_instruction.__name__}>"
        else:
            instruction = str(raw_instruction) if raw_instruction else ""

        model = agent.model if isinstance(agent.model, str) else str(agent.model) if agent.model else ""
        tools = _parse_tools_sync(agent.tools)
        output_key = agent.output_key
        disallow_transfer_to_parent = agent.disallow_transfer_to_parent
        disallow_transfer_to_peers = agent.disallow_transfer_to_peers

    sub_agents = [_parse_agent_recursive(sub) for sub in agent.sub_agents]

    return AgentMetadata(
        name=agent.name,
        agent_type=_get_agent_type(agent),
        description=agent.description or "",
        instruction=instruction,
        model=model,
        tools=tools,
        sub_agents=sub_agents,
        output_key=output_key,
        disallow_transfer_to_parent=disallow_transfer_to_parent,
        disallow_transfer_to_peers=disallow_transfer_to_peers,
    )


def parse_agent(
    agent: BaseAgent,
    save_path: Optional[str] = None,
) -> AgentMetadata:
    """Parse an ADK agent object into an AgentMetadata tree.

    Args:
        agent: A live ADK BaseAgent instance (Agent, SequentialAgent, etc.)
        save_path: Optional file path to save the metadata JSON.

    Returns:
        AgentMetadata with the full recursive structure.
    """
    metadata = _parse_agent_recursive(agent)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(metadata.model_dump_json(indent=2))

    return metadata
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_parser.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add adk_eval_tool/agent_parser/ tests/conftest.py tests/test_parser.py
git commit -m "feat: add agent parser to introspect ADK agents into metadata trees"
```

---

## Task 3: Agent Parser — MCP Tool Resolution

**Files:**
- Create: `adk_eval_tool/agent_parser/mcp_resolver.py`
- Create: `tests/test_mcp_resolver.py`
- Modify: `adk_eval_tool/agent_parser/parser.py` — add async `parse_agent_async` that resolves MCP toolsets

- [ ] **Step 1: Write MCP resolver tests**

```python
# tests/test_mcp_resolver.py
"""Tests for MCP tool resolution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from adk_eval_tool.agent_parser.mcp_resolver import resolve_mcp_toolset
from adk_eval_tool.schemas import ToolMetadata


@pytest.mark.asyncio
async def test_resolve_mcp_toolset_returns_tool_metadata():
    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.description = "Read a file from disk"
    mock_tool._get_declaration.return_value = None

    mock_toolset = AsyncMock()
    mock_toolset.get_tools = AsyncMock(return_value=[mock_tool])

    tools = await resolve_mcp_toolset(mock_toolset, server_name="filesystem")
    assert len(tools) == 1
    assert tools[0].name == "read_file"
    assert tools[0].source == "mcp"
    assert tools[0].mcp_server_name == "filesystem"


@pytest.mark.asyncio
async def test_resolve_mcp_toolset_handles_failure():
    mock_toolset = AsyncMock()
    mock_toolset.get_tools = AsyncMock(side_effect=Exception("Connection refused"))

    tools = await resolve_mcp_toolset(mock_toolset, server_name="broken_server")
    assert len(tools) == 1
    assert "unresolved" in tools[0].name.lower() or "error" in tools[0].description.lower()


@pytest.mark.asyncio
async def test_resolve_mcp_toolset_multiple_tools():
    mock_tools = []
    for name, desc in [("read", "Read file"), ("write", "Write file"), ("list", "List dir")]:
        t = MagicMock()
        t.name = name
        t.description = desc
        t._get_declaration.return_value = None
        mock_tools.append(t)

    mock_toolset = AsyncMock()
    mock_toolset.get_tools = AsyncMock(return_value=mock_tools)

    tools = await resolve_mcp_toolset(mock_toolset, server_name="fs")
    assert len(tools) == 3
    assert all(t.source == "mcp" for t in tools)
    assert all(t.mcp_server_name == "fs" for t in tools)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement MCP resolver**

```python
# adk_eval_tool/agent_parser/mcp_resolver.py
"""Resolve MCP toolset tools into ToolMetadata."""

from __future__ import annotations

from typing import Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset

from adk_eval_tool.schemas import ToolMetadata


async def resolve_mcp_toolset(
    toolset: BaseToolset,
    server_name: str = "unknown",
) -> list[ToolMetadata]:
    """Resolve all tools from an MCP toolset.

    Args:
        toolset: An MCPToolset or other BaseToolset instance.
        server_name: Name identifier for the MCP server.

    Returns:
        List of ToolMetadata with source="mcp".
    """
    try:
        tools: list[BaseTool] = await toolset.get_tools()
    except Exception as e:
        return [ToolMetadata(
            name=f"unresolved:{server_name}",
            description=f"Error resolving MCP tools: {e}",
            parameters_schema={},
            source="mcp",
            mcp_server_name=server_name,
        )]

    result = []
    for tool in tools:
        schema = _extract_tool_schema(tool)
        result.append(ToolMetadata(
            name=tool.name,
            description=tool.description or "",
            parameters_schema=schema,
            source="mcp",
            mcp_server_name=server_name,
        ))
    return result


def _extract_tool_schema(tool: BaseTool) -> dict:
    """Extract parameter schema from a BaseTool."""
    try:
        decl = tool._get_declaration()
        if decl and decl.parameters:
            schema = {"type": "object", "properties": {}}
            if hasattr(decl.parameters, "properties") and decl.parameters.properties:
                for prop_name, prop_schema in decl.parameters.properties.items():
                    schema["properties"][prop_name] = {
                        "type": getattr(prop_schema, "type", "string"),
                    }
            if hasattr(decl.parameters, "required") and decl.parameters.required:
                schema["required"] = list(decl.parameters.required)
            return schema
    except Exception:
        pass
    return {}
```

- [ ] **Step 4: Add async parse_agent_async to parser.py**

Add to the end of `adk_eval_tool/agent_parser/parser.py`:

```python
async def parse_agent_async(
    agent: BaseAgent,
    save_path: Optional[str] = None,
) -> AgentMetadata:
    """Parse an ADK agent, resolving MCP toolsets asynchronously.

    Same as parse_agent() but resolves MCPToolset tools by connecting
    to MCP servers and listing their available tools.

    Args:
        agent: A live ADK BaseAgent instance.
        save_path: Optional file path to save the metadata JSON.

    Returns:
        AgentMetadata with MCP tools fully resolved.
    """
    from adk_eval_tool.agent_parser.mcp_resolver import resolve_mcp_toolset

    metadata = await _parse_agent_recursive_async(agent)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(metadata.model_dump_json(indent=2))

    return metadata


async def _parse_agent_recursive_async(agent: BaseAgent) -> AgentMetadata:
    """Recursively parse, resolving MCP toolsets."""
    instruction = ""
    model = ""
    tools: list[ToolMetadata] = []
    output_key = None
    disallow_transfer_to_parent = False
    disallow_transfer_to_peers = False

    if isinstance(agent, LlmAgent):
        raw_instruction = agent.instruction
        if isinstance(raw_instruction, str):
            instruction = raw_instruction
        elif callable(raw_instruction):
            instruction = f"<dynamic: {raw_instruction.__name__}>"
        else:
            instruction = str(raw_instruction) if raw_instruction else ""

        model = agent.model if isinstance(agent.model, str) else str(agent.model) if agent.model else ""
        output_key = agent.output_key
        disallow_transfer_to_parent = agent.disallow_transfer_to_parent
        disallow_transfer_to_peers = agent.disallow_transfer_to_peers

        from adk_eval_tool.agent_parser.mcp_resolver import resolve_mcp_toolset

        for tool in agent.tools:
            if isinstance(tool, BaseToolset):
                # Attempt to get server name from toolset
                server_name = getattr(tool, "name", None) or type(tool).__name__
                resolved = await resolve_mcp_toolset(tool, server_name=server_name)
                tools.extend(resolved)
            else:
                tools.append(_parse_tool(tool))

    sub_agents = []
    for sub in agent.sub_agents:
        sub_agents.append(await _parse_agent_recursive_async(sub))

    return AgentMetadata(
        name=agent.name,
        agent_type=_get_agent_type(agent),
        description=agent.description or "",
        instruction=instruction,
        model=model,
        tools=tools,
        sub_agents=sub_agents,
        output_key=output_key,
        disallow_transfer_to_parent=disallow_transfer_to_parent,
        disallow_transfer_to_peers=disallow_transfer_to_peers,
    )
```

Update `adk_eval_tool/agent_parser/__init__.py`:
```python
from adk_eval_tool.agent_parser.parser import parse_agent, parse_agent_async

__all__ = ["parse_agent", "parse_agent_async"]
```

- [ ] **Step 5: Run all parser tests**

Run: `python -m pytest tests/test_parser.py tests/test_mcp_resolver.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add adk_eval_tool/agent_parser/ tests/test_mcp_resolver.py
git commit -m "feat: add MCP tool resolution and async agent parsing"
```

---

## Task 4: Intent & Scenario Generator Agent

**Files:**
- Create: `adk_eval_tool/intent_generator/__init__.py`
- Create: `adk_eval_tool/intent_generator/prompts.py`
- Create: `adk_eval_tool/intent_generator/tools.py`
- Create: `adk_eval_tool/intent_generator/agent.py`
- Create: `tests/test_intent_generator.py`

- [ ] **Step 1: Write intent generator tests**

```python
# tests/test_intent_generator.py
"""Tests for intent/scenario generator agent."""

import json
import pytest
from adk_eval_tool.intent_generator.prompts import build_system_instruction
from adk_eval_tool.intent_generator.tools import (
    validate_intent_output,
    format_agent_metadata_summary,
)
from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    IntentScenarioSet,
    Intent,
    Scenario,
    ScenarioStep,
)


def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="travel_agent",
        agent_type="LlmAgent",
        description="Helps users book flights and hotels",
        instruction="You are a travel assistant.",
        model="gemini-2.0-flash",
        tools=[
            ToolMetadata(name="search_flights", description="Search for flights", parameters_schema={}),
            ToolMetadata(name="book_flight", description="Book a flight", parameters_schema={}),
            ToolMetadata(name="search_hotels", description="Search for hotels", parameters_schema={}),
        ],
        sub_agents=[
            AgentMetadata(
                name="payment_agent",
                agent_type="LlmAgent",
                description="Handles payments",
                instruction="Process payments.",
                model="gemini-2.0-flash",
                tools=[ToolMetadata(name="process_payment", description="Process payment", parameters_schema={})],
                sub_agents=[],
            )
        ],
    )


def test_build_system_instruction_contains_metadata():
    metadata = _sample_metadata()
    instruction = build_system_instruction(metadata)
    assert "travel_agent" in instruction
    assert "search_flights" in instruction
    assert "payment_agent" in instruction


def test_format_agent_metadata_summary():
    metadata = _sample_metadata()
    summary = format_agent_metadata_summary(metadata)
    assert "travel_agent" in summary
    assert "search_flights" in summary
    assert "payment_agent" in summary


def test_validate_intent_output_valid():
    intent_set = IntentScenarioSet(
        agent_name="travel_agent",
        intents=[
            Intent(
                intent_id="book_flight",
                name="Book Flight",
                description="User wants to book a flight",
                scenarios=[
                    Scenario(
                        scenario_id="happy_path",
                        name="Successful booking",
                        steps=[
                            ScenarioStep(
                                user_message="Book a flight to London",
                                expected_tool_calls=["search_flights"],
                            )
                        ],
                    )
                ],
            )
        ],
    )
    result = validate_intent_output(intent_set.model_dump())
    assert result["valid"] is True


def test_validate_intent_output_invalid():
    result = validate_intent_output({"bad": "data"})
    assert result["valid"] is False
    assert "errors" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_intent_generator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement prompts.py**

```python
# adk_eval_tool/intent_generator/prompts.py
"""Prompt templates for the intent/scenario generator agent."""

from __future__ import annotations

from adk_eval_tool.schemas import AgentMetadata


def _format_tools(metadata: AgentMetadata, indent: int = 0) -> str:
    """Format tool list for prompt."""
    prefix = "  " * indent
    lines = []
    for tool in metadata.tools:
        lines.append(f"{prefix}- {tool.name}: {tool.description}")
        if tool.parameters_schema.get("properties"):
            for param, info in tool.parameters_schema["properties"].items():
                lines.append(f"{prefix}  param: {param} ({info.get('type', 'any')})")
    return "\n".join(lines)


def _format_agent_tree(metadata: AgentMetadata, indent: int = 0) -> str:
    """Format agent tree recursively."""
    prefix = "  " * indent
    lines = [
        f"{prefix}Agent: {metadata.name} ({metadata.agent_type})",
        f"{prefix}  Description: {metadata.description}",
        f"{prefix}  Instruction: {metadata.instruction[:200]}{'...' if len(metadata.instruction) > 200 else ''}",
        f"{prefix}  Model: {metadata.model}",
        f"{prefix}  Tools:",
    ]
    if metadata.tools:
        lines.append(_format_tools(metadata, indent + 2))
    else:
        lines.append(f"{prefix}    (none)")

    if metadata.sub_agents:
        lines.append(f"{prefix}  Sub-agents:")
        for sub in metadata.sub_agents:
            lines.append(_format_agent_tree(sub, indent + 2))

    return "\n".join(lines)


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert test designer for AI agents. Your job is to analyze an agent's capabilities and generate comprehensive test intents and scenarios.

## Agent Under Test

{agent_tree}

## Your Task

Given the agent metadata above and any user constraints, generate a comprehensive set of intents (user goals) and for each intent, generate detailed test scenarios.

## Output Requirements

You MUST output valid JSON matching this exact structure:

```json
{{
  "agent_name": "string",
  "intents": [
    {{
      "intent_id": "string (snake_case)",
      "name": "string",
      "description": "string",
      "category": "string",
      "scenarios": [
        {{
          "scenario_id": "string (snake_case)",
          "name": "string",
          "description": "string",
          "steps": [
            {{
              "user_message": "string (the actual user message)",
              "expected_tool_calls": ["tool_name_1", "tool_name_2"],
              "expected_tool_args": {{"tool_name": {{"arg": "value"}}}},
              "expected_response_keywords": ["keyword1", "keyword2"],
              "notes": "string"
            }}
          ],
          "tags": ["happy_path" | "edge_case" | "error" | "multi_turn"]
        }}
      ]
    }}
  ],
  "generation_context": "string"
}}
```

## Guidelines

1. **Intent Coverage**: Identify ALL distinct user intents the agent can handle, including:
   - Primary happy-path intents
   - Edge cases (missing info, ambiguous input, boundary values)
   - Error scenarios (invalid input, tool failures, unauthorized access)
   - Multi-turn conversations requiring clarification

2. **Scenario Design**: For each intent, create scenarios that:
   - Exercise different tool combinations
   - Cover sub-agent delegation paths
   - Test the full trajectory (which tools in what order)
   - Include realistic user messages

3. **Tool Coverage**: Ensure every tool and sub-agent is exercised by at least one scenario.

4. Use the `save_output` tool to save your final output.
"""


def build_system_instruction(metadata: AgentMetadata) -> str:
    """Build the system instruction with agent metadata embedded."""
    agent_tree = _format_agent_tree(metadata)
    return SYSTEM_INSTRUCTION_TEMPLATE.format(agent_tree=agent_tree)
```

- [ ] **Step 4: Implement tools.py**

```python
# adk_eval_tool/intent_generator/tools.py
"""Function tools for the intent/scenario generator agent."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from adk_eval_tool.schemas import AgentMetadata, IntentScenarioSet


def format_agent_metadata_summary(metadata: AgentMetadata) -> str:
    """Format agent metadata into a readable summary for the LLM.

    Args:
        metadata: The agent metadata to summarize.

    Returns:
        A human-readable summary string.
    """
    lines = [f"Agent: {metadata.name} ({metadata.agent_type})"]
    lines.append(f"Description: {metadata.description}")
    lines.append(f"Tools: {', '.join(t.name for t in metadata.tools)}")
    if metadata.sub_agents:
        lines.append(f"Sub-agents: {', '.join(a.name for a in metadata.sub_agents)}")
        for sub in metadata.sub_agents:
            lines.append(f"  - {sub.name}: {sub.description} [tools: {', '.join(t.name for t in sub.tools)}]")
    return "\n".join(lines)


def validate_intent_output(data: dict[str, Any]) -> dict[str, Any]:
    """Validate that generated output matches IntentScenarioSet schema.

    Args:
        data: The generated intent/scenario data as a dict.

    Returns:
        Dict with 'valid' bool and optional 'errors' list.
    """
    try:
        IntentScenarioSet.model_validate(data)
        return {"valid": True, "errors": []}
    except ValidationError as e:
        return {
            "valid": False,
            "errors": [str(err) for err in e.errors()],
        }


def save_output(output_json: str) -> str:
    """Save the final intent/scenario output. Called by the agent when done.

    Args:
        output_json: The complete JSON string of the IntentScenarioSet.

    Returns:
        Validation result message.
    """
    try:
        data = json.loads(output_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    validation = validate_intent_output(data)
    if validation["valid"]:
        return "Output validated successfully."
    else:
        return f"Validation errors: {validation['errors']}"
```

- [ ] **Step 5: Implement agent.py**

```python
# adk_eval_tool/intent_generator/__init__.py
"""Intent and scenario generator agent."""

from adk_eval_tool.intent_generator.agent import generate_intents

__all__ = ["generate_intents"]
```

```python
# adk_eval_tool/intent_generator/agent.py
"""ADK agent that generates intents and scenarios from agent metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_eval_tool.schemas import AgentMetadata, IntentScenarioSet
from adk_eval_tool.intent_generator.prompts import build_system_instruction
from adk_eval_tool.intent_generator.tools import save_output, validate_intent_output


def _create_intent_generator_agent(metadata: AgentMetadata) -> Agent:
    """Create an ADK agent configured for intent/scenario generation."""
    return Agent(
        name="intent_generator",
        model="gemini-2.0-flash",
        description="Generates test intents and scenarios for an agent",
        instruction=build_system_instruction(metadata),
        tools=[save_output],
    )


async def generate_intents(
    metadata: AgentMetadata,
    user_constraints: str = "",
    num_scenarios_per_intent: int = 3,
    save_path: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> IntentScenarioSet:
    """Generate intents and scenarios for an agent using an ADK agent.

    Args:
        metadata: Parsed agent metadata.
        user_constraints: Additional context or constraints from the user.
        num_scenarios_per_intent: Target number of scenarios per intent.
        save_path: Optional file path to save the output JSON.
        model: Model to use for generation.

    Returns:
        IntentScenarioSet with generated intents and scenarios.
    """
    agent = _create_intent_generator_agent(metadata)
    if model != "gemini-2.0-flash":
        agent.model = model

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="intent_gen", session_service=session_service)

    user_message = f"""Generate comprehensive test intents and scenarios for the agent described in your instructions.

Requirements:
- Generate {num_scenarios_per_intent} scenarios per intent (minimum)
- Cover happy paths, edge cases, and error scenarios
- Every tool and sub-agent must appear in at least one scenario
"""
    if user_constraints:
        user_message += f"\nAdditional constraints:\n{user_constraints}"

    user_message += "\n\nCall the save_output tool with the complete JSON when done."

    session = await session_service.create_session(app_name="intent_gen", user_id="generator")

    content = types.Content(role="user", parts=[types.Part(text=user_message)])

    final_text = ""
    async for event in runner.run_async(
        user_id="generator",
        session_id=session.id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text = part.text

    # Try to extract JSON from agent output or tool call
    intent_set = _extract_intent_set(final_text, metadata.name)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(intent_set.model_dump_json(indent=2))

    return intent_set


def _extract_intent_set(text: str, agent_name: str) -> IntentScenarioSet:
    """Extract IntentScenarioSet from agent output text."""
    # Try to find JSON block in the text
    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "agent_name" not in data:
                data["agent_name"] = agent_name
            return IntentScenarioSet.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    # Fallback: return empty set
    return IntentScenarioSet(agent_name=agent_name, intents=[], generation_context="Generation failed to produce valid JSON")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_intent_generator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add adk_eval_tool/intent_generator/ tests/test_intent_generator.py
git commit -m "feat: add intent/scenario generator ADK agent"
```

---

## Task 5: Test Case Generator Agent

**Files:**
- Create: `adk_eval_tool/testcase_generator/__init__.py`
- Create: `adk_eval_tool/testcase_generator/prompts.py`
- Create: `adk_eval_tool/testcase_generator/tools.py`
- Create: `adk_eval_tool/testcase_generator/agent.py`
- Create: `tests/test_testcase_generator.py`

- [ ] **Step 1: Write test case generator tests**

```python
# tests/test_testcase_generator.py
"""Tests for test case generator."""

import json
import pytest
from adk_eval_tool.testcase_generator.tools import (
    build_eval_case_json,
    build_eval_set_json,
    validate_eval_set,
)
from adk_eval_tool.testcase_generator.prompts import build_testcase_system_instruction
from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    Intent,
    Scenario,
    ScenarioStep,
    TestCaseConfig,
)


def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="travel_agent",
        agent_type="LlmAgent",
        description="Helps book flights",
        instruction="You are a travel assistant.",
        model="gemini-2.0-flash",
        tools=[
            ToolMetadata(name="search_flights", description="Search flights", parameters_schema={}),
        ],
        sub_agents=[],
    )


def _sample_intent() -> Intent:
    return Intent(
        intent_id="book_flight",
        name="Book Flight",
        description="User books a flight",
        category="booking",
        scenarios=[
            Scenario(
                scenario_id="happy_path",
                name="Successful booking",
                steps=[
                    ScenarioStep(
                        user_message="Find flights to London tomorrow",
                        expected_tool_calls=["search_flights"],
                        expected_tool_args={"search_flights": {"destination": "London"}},
                        tool_responses={"search_flights": {"flights": [{"id": "FL-1"}]}},
                        expected_response="I found flights to London.",
                        expected_response_keywords=["London", "flight"],
                        rubric="Agent should present flight options clearly.",
                    ),
                    ScenarioStep(
                        user_message="Book the cheapest one",
                        expected_tool_calls=["search_flights"],
                        expected_response_keywords=["booked"],
                    ),
                ],
            )
        ],
    )


def test_build_eval_case_json_structure():
    scenario = _sample_intent().scenarios[0]
    eval_case = build_eval_case_json(
        scenario=scenario.model_dump(),
        intent_id="book_flight",
    )
    assert eval_case["evalId"] == "book_flight__happy_path"
    assert "conversation" in eval_case
    assert len(eval_case["conversation"]) == 2
    inv = eval_case["conversation"][0]
    assert inv["userContent"]["parts"][0]["text"] == "Find flights to London tomorrow"
    assert inv["intermediateData"]["toolUses"][0]["name"] == "search_flights"
    assert inv["intermediateData"]["toolUses"][0]["args"]["destination"] == "London"
    assert len(inv["intermediateData"]["toolResponses"]) == 1
    assert inv["finalResponse"]["parts"][0]["text"] == "I found flights to London."
    assert "rubrics" in inv
    assert inv["rubrics"][0]["rubricId"] == "rubric_inv_1"


def test_build_eval_case_conversation_scenario():
    scenario = Scenario(
        scenario_id="dynamic_test",
        name="Dynamic test",
        eval_type="conversation_scenario",
        conversation_scenario={
            "starting_prompt": "I need help booking",
            "conversation_plan": "Ask for destination. Confirm booking.",
        },
        session_state={"user_id": "test_123"},
    )
    eval_case = build_eval_case_json(
        scenario=scenario.model_dump(),
        intent_id="book_flight",
    )
    assert eval_case["evalId"] == "book_flight__dynamic_test"
    assert "conversation_scenario" in eval_case
    assert "conversation" not in eval_case
    assert eval_case["sessionInput"]["state"]["user_id"] == "test_123"


def test_build_eval_set_json():
    intent = _sample_intent()
    eval_set = build_eval_set_json(
        intent=intent.model_dump(),
        agent_name="travel_agent",
    )
    assert eval_set["evalSetId"] == "travel_agent__book_flight"
    assert len(eval_set["evalCases"]) == 1


def test_validate_eval_set_valid():
    intent = _sample_intent()
    eval_set = build_eval_set_json(intent=intent.model_dump(), agent_name="travel_agent")
    result = validate_eval_set(json.dumps(eval_set))
    assert result["valid"] is True


def test_build_testcase_system_instruction():
    metadata = _sample_metadata()
    config = TestCaseConfig()
    instruction = build_testcase_system_instruction(metadata, config)
    assert "travel_agent" in instruction
    assert "evalset.json" in instruction.lower() or "EvalSet" in instruction
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_testcase_generator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement prompts.py**

```python
# adk_eval_tool/testcase_generator/prompts.py
"""Prompt templates for the test case generator agent."""

from __future__ import annotations

from adk_eval_tool.intent_generator.prompts import _format_agent_tree
from adk_eval_tool.schemas import AgentMetadata, TestCaseConfig


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert at creating evaluation datasets for Google ADK agents. You generate golden test cases in the ADK `.evalset.json` format.

## Agent Under Test

{agent_tree}

## Evaluation Config

- Metrics: {metrics}
- Tool trajectory match type: {match_type}
- Judge model: {judge_model}

## ADK EvalSet JSON Format

You MUST produce JSON in this exact structure (camelCase keys):

```json
{{
  "evalSetId": "<agent_name>__<intent_id>",
  "name": "<Human-readable name>",
  "description": "<What this eval set tests>",
  "evalCases": [
    {{
      "evalId": "<intent_id>__<scenario_id>",
      "conversation": [
        {{
          "invocationId": "inv-<n>",
          "userContent": {{
            "role": "user",
            "parts": [{{"text": "<user message>"}}]
          }},
          "finalResponse": {{
            "role": "model",
            "parts": [{{"text": "<expected response summary>"}}]
          }},
          "intermediateData": {{
            "toolUses": [
              {{"name": "<tool_name>", "args": {{<expected_args>}}}}
            ],
            "toolResponses": [],
            "intermediateResponses": []
          }}
        }}
      ]
    }}
  ]
}}
```

## Important Rules

1. Each conversation turn maps to one Invocation object
2. `toolUses` must list the tools the agent SHOULD call in order
3. `finalResponse` should describe what a correct response looks like
4. `evalId` format: `<intent_id>__<scenario_id>`
5. `evalSetId` format: `<agent_name>__<intent_id>`
6. Multi-turn scenarios have multiple entries in the `conversation` array

Use the `save_eval_set` tool to save your output. Use the `validate_eval_set` tool to check your JSON before saving.
"""


def build_testcase_system_instruction(
    metadata: AgentMetadata,
    config: TestCaseConfig,
) -> str:
    """Build system instruction for the test case generator."""
    agent_tree = _format_agent_tree(metadata)
    metrics = ", ".join(f"{k}={v}" for k, v in config.eval_metrics.items())
    return SYSTEM_INSTRUCTION_TEMPLATE.format(
        agent_tree=agent_tree,
        metrics=metrics,
        match_type=config.tool_trajectory_match_type,
        judge_model=config.judge_model,
    )
```

- [ ] **Step 4: Implement tools.py**

```python
# adk_eval_tool/testcase_generator/tools.py
"""Function tools for the test case generator agent."""

from __future__ import annotations

import json
import uuid
from typing import Any


def build_eval_case_json(
    scenario: dict[str, Any],
    intent_id: str,
) -> dict[str, Any]:
    """Build an ADK EvalCase JSON object from a scenario.

    Handles both trajectory-based (static) and conversation_scenario (dynamic)
    eval types. For trajectory-based scenarios, maps ScenarioStep fields to ADK
    Invocation structure including toolUses, toolResponses, intermediateResponses,
    finalResponse, and per-invocation rubrics.

    Args:
        scenario: A scenario dict with scenario_id, steps, eval_type, etc.
        intent_id: The parent intent ID.

    Returns:
        An EvalCase dict in ADK camelCase format.
    """
    scenario_id = scenario.get("scenario_id", "unknown")
    eval_id = f"{intent_id}__{scenario_id}"
    eval_type = scenario.get("eval_type", "trajectory")

    # Dynamic conversation_scenario mode
    if eval_type == "conversation_scenario" and scenario.get("conversation_scenario"):
        eval_case: dict[str, Any] = {
            "evalId": eval_id,
            "conversation_scenario": scenario["conversation_scenario"],
        }
        if scenario.get("session_state"):
            eval_case["sessionInput"] = {
                "appName": "eval_app",
                "userId": "test_user",
                "state": scenario["session_state"],
            }
        return eval_case

    # Static trajectory mode
    conversation = []
    for i, step in enumerate(scenario.get("steps", [])):
        # Build toolUses with args
        tool_uses = []
        for tool_name in step.get("expected_tool_calls", []):
            tool_use = {"name": tool_name}
            tool_args = step.get("expected_tool_args", {})
            if tool_args and tool_name in tool_args:
                tool_use["args"] = tool_args[tool_name]
            else:
                tool_use["args"] = {}
            tool_uses.append(tool_use)

        # Build toolResponses (reference data for hallucination eval)
        tool_responses = []
        if step.get("tool_responses"):
            for tool_name, response_data in step["tool_responses"].items():
                tool_responses.append({
                    "name": tool_name,
                    "id": f"call_{i}_{tool_name}",
                    "response": response_data,
                })

        # Build intermediateResponses
        intermediate_responses = step.get("intermediate_responses", [])

        invocation: dict[str, Any] = {
            "invocationId": f"inv-{i + 1}",
            "userContent": {
                "role": "user",
                "parts": [{"text": step.get("user_message", "")}],
            },
            "intermediateData": {
                "toolUses": tool_uses,
                "toolResponses": tool_responses,
                "intermediateResponses": intermediate_responses,
            },
        }

        # Add finalResponse: prefer explicit expected_response, fall back to keywords
        expected_response = step.get("expected_response", "")
        keywords = step.get("expected_response_keywords", [])
        if expected_response:
            invocation["finalResponse"] = {
                "role": "model",
                "parts": [{"text": expected_response}],
            }
        elif keywords:
            invocation["finalResponse"] = {
                "role": "model",
                "parts": [{"text": f"Response should include: {', '.join(keywords)}"}],
            }

        # Add per-invocation rubric if present
        if step.get("rubric"):
            invocation["rubrics"] = [{
                "rubricId": f"rubric_inv_{i + 1}",
                "rubricContent": {"textProperty": step["rubric"]},
            }]

        conversation.append(invocation)

    eval_case = {
        "evalId": eval_id,
        "conversation": conversation,
    }

    # Add session state if present
    if scenario.get("session_state"):
        eval_case["sessionInput"] = {
            "appName": "eval_app",
            "userId": "test_user",
            "state": scenario["session_state"],
        }

    return eval_case


def build_eval_set_json(
    intent: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    """Build a complete ADK EvalSet JSON from an intent.

    Args:
        intent: An intent dict with intent_id, scenarios, etc.
        agent_name: The name of the agent being tested.

    Returns:
        An EvalSet dict in ADK camelCase format.
    """
    intent_id = intent.get("intent_id", "unknown")
    eval_cases = []
    for scenario in intent.get("scenarios", []):
        eval_cases.append(build_eval_case_json(scenario, intent_id))

    return {
        "evalSetId": f"{agent_name}__{intent_id}",
        "name": intent.get("name", intent_id),
        "description": intent.get("description", ""),
        "evalCases": eval_cases,
    }


def validate_eval_set(eval_set_json: str) -> dict[str, Any]:
    """Validate that a JSON string is a valid ADK EvalSet structure.

    Args:
        eval_set_json: JSON string of an EvalSet.

    Returns:
        Dict with 'valid' bool and optional 'errors' list.
    """
    errors = []
    try:
        data = json.loads(eval_set_json)
    except json.JSONDecodeError as e:
        return {"valid": False, "errors": [f"Invalid JSON: {e}"]}

    if "evalSetId" not in data:
        errors.append("Missing 'evalSetId'")
    if "evalCases" not in data:
        errors.append("Missing 'evalCases'")
    elif not isinstance(data["evalCases"], list):
        errors.append("'evalCases' must be a list")
    else:
        for i, case in enumerate(data["evalCases"]):
            if "evalId" not in case:
                errors.append(f"evalCases[{i}]: missing 'evalId'")
            if "conversation" not in case and "conversation_scenario" not in case:
                errors.append(f"evalCases[{i}]: must have 'conversation' or 'conversation_scenario'")
            if "conversation" in case:
                for j, inv in enumerate(case["conversation"]):
                    if "userContent" not in inv:
                        errors.append(f"evalCases[{i}].conversation[{j}]: missing 'userContent'")

    return {"valid": len(errors) == 0, "errors": errors}


def save_eval_set(eval_set_json: str) -> str:
    """Save and validate a generated EvalSet JSON. Called by the agent.

    Args:
        eval_set_json: The complete EvalSet JSON string.

    Returns:
        Validation result message.
    """
    result = validate_eval_set(eval_set_json)
    if result["valid"]:
        return "EvalSet validated successfully and ready to save."
    else:
        return f"Validation errors: {result['errors']}"
```

- [ ] **Step 5: Implement agent.py**

```python
# adk_eval_tool/testcase_generator/__init__.py
"""Test case generator agent."""

from adk_eval_tool.testcase_generator.agent import generate_test_cases

__all__ = ["generate_test_cases"]
```

```python
# adk_eval_tool/testcase_generator/agent.py
"""ADK agent that generates golden test cases in ADK evalset format."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_eval_tool.schemas import (
    AgentMetadata,
    Intent,
    IntentScenarioSet,
    TestCaseConfig,
)
from adk_eval_tool.testcase_generator.prompts import build_testcase_system_instruction
from adk_eval_tool.testcase_generator.tools import (
    validate_eval_set,
    save_eval_set,
    build_eval_set_json,
)


def _create_testcase_generator_agent(
    metadata: AgentMetadata,
    config: TestCaseConfig,
) -> Agent:
    """Create an ADK agent for test case generation."""
    return Agent(
        name="testcase_generator",
        model=config.judge_model,
        description="Generates ADK-compatible evaluation test cases",
        instruction=build_testcase_system_instruction(metadata, config),
        tools=[validate_eval_set, save_eval_set],
    )


async def generate_test_cases(
    metadata: AgentMetadata,
    intent: Intent,
    config: Optional[TestCaseConfig] = None,
    save_dir: Optional[str] = None,
) -> dict:
    """Generate an ADK EvalSet for a single intent.

    Args:
        metadata: Agent metadata.
        intent: The intent to generate test cases for.
        config: Evaluation config (metrics, thresholds, judge model).
        save_dir: Optional directory to save the .evalset.json file.

    Returns:
        Dict in ADK EvalSet JSON format (camelCase).
    """
    config = config or TestCaseConfig()
    agent = _create_testcase_generator_agent(metadata, config)

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="testcase_gen", session_service=session_service)

    prompt = f"""Generate an ADK EvalSet for this intent:

Intent: {intent.name} ({intent.intent_id})
Description: {intent.description}
Category: {intent.category}

Scenarios:
{json.dumps([s.model_dump() for s in intent.scenarios], indent=2)}

Generate the complete evalset.json. Call validate_eval_set to check your output, then call save_eval_set with the final JSON.
"""

    session = await session_service.create_session(app_name="testcase_gen", user_id="generator")
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = ""
    async for event in runner.run_async(
        user_id="generator",
        session_id=session.id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text = part.text

    # Extract JSON from output
    eval_set = _extract_eval_set(final_text, metadata.name, intent)

    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{metadata.name}__{intent.intent_id}.evalset.json"
        (path / filename).write_text(json.dumps(eval_set, indent=2))

    return eval_set


def _extract_eval_set(text: str, agent_name: str, intent: Intent) -> dict:
    """Extract EvalSet JSON from agent output, with fallback to programmatic build."""
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            validation = validate_eval_set(json.dumps(data))
            if validation["valid"]:
                return data
        except (json.JSONDecodeError, Exception):
            pass

    # Fallback: build from scenario data programmatically
    return build_eval_set_json(intent.model_dump(), agent_name)


async def generate_all_test_cases(
    metadata: AgentMetadata,
    intent_set: IntentScenarioSet,
    config: Optional[TestCaseConfig] = None,
    save_dir: Optional[str] = None,
) -> list[dict]:
    """Generate EvalSets for all intents in an IntentScenarioSet.

    Args:
        metadata: Agent metadata.
        intent_set: All intents and scenarios.
        config: Evaluation config.
        save_dir: Optional directory to save all .evalset.json files.

    Returns:
        List of EvalSet dicts.
    """
    results = []
    for intent in intent_set.intents:
        eval_set = await generate_test_cases(metadata, intent, config, save_dir)
        results.append(eval_set)
    return results
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_testcase_generator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add adk_eval_tool/testcase_generator/ tests/test_testcase_generator.py
git commit -m "feat: add test case generator agent producing ADK evalset.json files"
```

---

## Task 6: Streamlit UI — App Shell & Navigation

**Files:**
- Create: `adk_eval_tool/ui/__init__.py`
- Create: `adk_eval_tool/ui/app.py`
- Create: `adk_eval_tool/ui/components/__init__.py`
- Create: `adk_eval_tool/ui/components/json_editor.py`
- Create: `adk_eval_tool/ui/components/run_status.py`
- Create: `adk_eval_tool/ui/pages/__init__.py`

- [ ] **Step 1: Implement JSON editor component**

```python
# adk_eval_tool/ui/components/__init__.py
"""Reusable Streamlit components."""
```

```python
# adk_eval_tool/ui/components/json_editor.py
"""JSON tree editor widget for Streamlit."""

from __future__ import annotations

import json
from typing import Any, Optional

import streamlit as st


def json_editor(
    data: dict[str, Any],
    key: str,
    readonly: bool = False,
) -> dict[str, Any]:
    """Render an editable JSON tree.

    Args:
        data: The JSON data to display/edit.
        key: Unique Streamlit key prefix.
        readonly: If True, display only.

    Returns:
        The edited (or original) data.
    """
    json_str = json.dumps(data, indent=2)

    if readonly:
        st.code(json_str, language="json")
        return data

    edited_str = st.text_area(
        "Edit JSON",
        value=json_str,
        height=400,
        key=f"{key}_editor",
    )

    try:
        edited_data = json.loads(edited_str)
        return edited_data
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return data
```

```python
# adk_eval_tool/ui/components/run_status.py
"""Generation progress/status display component."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def run_status_display(
    status: str,
    progress: float = 0.0,
    message: str = "",
    details: Optional[dict] = None,
):
    """Display generation run status.

    Args:
        status: One of "idle", "running", "completed", "failed".
        progress: Progress fraction 0.0 to 1.0.
        message: Status message.
        details: Optional details dict.
    """
    if status == "idle":
        st.info("Ready to run.")
    elif status == "running":
        st.progress(progress, text=message)
    elif status == "completed":
        st.success(message or "Completed successfully.")
    elif status == "failed":
        st.error(message or "Generation failed.")

    if details:
        with st.expander("Details"):
            st.json(details)
```

- [ ] **Step 2: Implement app shell**

```python
# adk_eval_tool/ui/__init__.py
"""Streamlit UI for ADK Eval Tool."""
```

```python
# adk_eval_tool/ui/pages/__init__.py
"""Streamlit pages."""
```

```python
# adk_eval_tool/ui/app.py
"""Main Streamlit application."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ADK Eval Tool",
    page_icon="🧪",
    layout="wide",
)


def main():
    st.sidebar.title("ADK Eval Tool")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Agent Metadata",
            "Intents & Scenarios",
            "Test Cases",
            "Eval Config",
            "Run Evaluation",
            "Eval Results",
            "Dataset Versions",
        ],
    )

    # Session state initialization
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "intent_set" not in st.session_state:
        st.session_state.intent_set = None
    if "eval_sets" not in st.session_state:
        st.session_state.eval_sets = []
    if "eval_run_config" not in st.session_state:
        st.session_state.eval_run_config = None
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []

    if page == "Agent Metadata":
        from adk_eval_tool.ui.pages.metadata_viewer import render
        render()
    elif page == "Intents & Scenarios":
        from adk_eval_tool.ui.pages.intent_manager import render
        render()
    elif page == "Test Cases":
        from adk_eval_tool.ui.pages.testcase_manager import render
        render()
    elif page == "Eval Config":
        from adk_eval_tool.ui.pages.eval_config import render
        render()
    elif page == "Run Evaluation":
        from adk_eval_tool.ui.pages.eval_launcher import render
        render()
    elif page == "Eval Results":
        from adk_eval_tool.ui.pages.eval_results import render
        render()
    elif page == "Dataset Versions":
        from adk_eval_tool.ui.pages.dataset_versions import render
        render()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add adk_eval_tool/ui/
git commit -m "feat: add Streamlit app shell, navigation, and reusable components"
```

---

## Task 7: Streamlit UI — Metadata Viewer Page

**Files:**
- Create: `adk_eval_tool/ui/pages/metadata_viewer.py`

- [ ] **Step 1: Implement metadata viewer page**

```python
# adk_eval_tool/ui/pages/metadata_viewer.py
"""Page: Parse agent, view, and edit metadata."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import AgentMetadata
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Agent Metadata")

    tab_load, tab_edit = st.tabs(["Load / Parse", "View / Edit"])

    with tab_load:
        st.subheader("Load from file")
        uploaded = st.file_uploader("Upload metadata JSON", type=["json"], key="meta_upload")
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                st.session_state.metadata = AgentMetadata.model_validate(data)
                st.success(f"Loaded metadata for agent: {st.session_state.metadata.name}")
            except Exception as e:
                st.error(f"Failed to load: {e}")

        st.divider()
        st.subheader("Parse from agent module")
        st.markdown("""
        To parse a live agent, provide the module path (e.g., `my_agent.agent`).
        The module must define a `root_agent` or named agent variable.
        """)

        col1, col2 = st.columns(2)
        with col1:
            module_path = st.text_input("Agent module path", placeholder="my_agent.agent")
        with col2:
            agent_var = st.text_input("Agent variable name", value="root_agent")

        if st.button("Parse Agent", disabled=not module_path):
            try:
                import importlib
                mod = importlib.import_module(module_path)
                agent_obj = getattr(mod, agent_var)

                from adk_eval_tool.agent_parser import parse_agent
                st.session_state.metadata = parse_agent(agent_obj)
                st.success(f"Parsed agent: {st.session_state.metadata.name}")
            except Exception as e:
                st.error(f"Failed to parse: {e}")

    with tab_edit:
        if st.session_state.metadata is None:
            st.warning("No metadata loaded. Use the 'Load / Parse' tab first.")
            return

        metadata = st.session_state.metadata

        # Agent tree visualization
        st.subheader("Agent Tree")
        _render_agent_tree(metadata)

        st.divider()

        # Editable JSON
        st.subheader("Edit Metadata")
        edited = json_editor(metadata.model_dump(), key="metadata")
        if edited != metadata.model_dump():
            try:
                st.session_state.metadata = AgentMetadata.model_validate(edited)
                st.success("Metadata updated.")
            except Exception as e:
                st.error(f"Invalid metadata: {e}")

        # Save
        col_save, col_download = st.columns(2)
        with col_save:
            save_path = st.text_input("Save path", value=f"{metadata.name}_metadata.json")
            if st.button("Save to disk"):
                Path(save_path).write_text(
                    st.session_state.metadata.model_dump_json(indent=2)
                )
                st.success(f"Saved to {save_path}")
        with col_download:
            st.download_button(
                "Download JSON",
                data=st.session_state.metadata.model_dump_json(indent=2),
                file_name=f"{metadata.name}_metadata.json",
                mime="application/json",
            )


def _render_agent_tree(metadata: AgentMetadata, level: int = 0):
    """Render agent tree using Streamlit expanders."""
    prefix = "  " * level
    with st.expander(f"{'📦' if level == 0 else '📎'} {metadata.name} ({metadata.agent_type})", expanded=(level == 0)):
        st.markdown(f"**Description:** {metadata.description}")
        st.markdown(f"**Model:** {metadata.model}")
        if metadata.instruction:
            st.markdown(f"**Instruction:** {metadata.instruction[:300]}{'...' if len(metadata.instruction) > 300 else ''}")
        if metadata.tools:
            st.markdown("**Tools:**")
            for tool in metadata.tools:
                st.markdown(f"- `{tool.name}` ({tool.source}): {tool.description}")
        if metadata.sub_agents:
            st.markdown("**Sub-agents:**")
            for sub in metadata.sub_agents:
                _render_agent_tree(sub, level + 1)
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/metadata_viewer.py
git commit -m "feat: add metadata viewer/editor Streamlit page"
```

---

## Task 8: Streamlit UI — Intent Manager Page

**Files:**
- Create: `adk_eval_tool/ui/pages/intent_manager.py`

- [ ] **Step 1: Implement intent manager page**

```python
# adk_eval_tool/ui/pages/intent_manager.py
"""Page: Generate, view, and edit intents & scenarios."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import IntentScenarioSet, Intent
from adk_eval_tool.ui.components.json_editor import json_editor
from adk_eval_tool.ui.components.run_status import run_status_display


def render():
    st.header("Intents & Scenarios")

    if st.session_state.metadata is None:
        st.warning("Load agent metadata first (Agent Metadata page).")
        return

    tab_gen, tab_edit = st.tabs(["Generate", "View / Edit"])

    with tab_gen:
        st.subheader("Generate Intents & Scenarios")

        constraints = st.text_area(
            "User constraints / context",
            placeholder="e.g., Focus on error handling scenarios, include multi-language inputs...",
            key="intent_constraints",
        )
        num_scenarios = st.slider("Scenarios per intent", min_value=1, max_value=10, value=3)
        model = st.selectbox("Generator model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])

        # Option to regenerate for specific intents
        if st.session_state.intent_set and st.session_state.intent_set.intents:
            st.divider()
            st.markdown("**Selective regeneration:**")
            existing_intents = [i.intent_id for i in st.session_state.intent_set.intents]
            selected_intents = st.multiselect(
                "Regenerate only these intents (leave empty for full generation)",
                options=existing_intents,
            )
        else:
            selected_intents = []

        if st.button("Generate", key="gen_intents"):
            with st.spinner("Generating intents and scenarios..."):
                from adk_eval_tool.intent_generator import generate_intents

                try:
                    result = asyncio.run(generate_intents(
                        metadata=st.session_state.metadata,
                        user_constraints=constraints,
                        num_scenarios_per_intent=num_scenarios,
                        model=model,
                    ))

                    if selected_intents and st.session_state.intent_set:
                        # Merge: replace only selected intents
                        existing = st.session_state.intent_set
                        new_map = {i.intent_id: i for i in result.intents}
                        merged_intents = []
                        for intent in existing.intents:
                            if intent.intent_id in selected_intents and intent.intent_id in new_map:
                                merged_intents.append(new_map[intent.intent_id])
                            else:
                                merged_intents.append(intent)
                        # Add any new intents not in existing
                        existing_ids = {i.intent_id for i in existing.intents}
                        for intent in result.intents:
                            if intent.intent_id not in existing_ids:
                                merged_intents.append(intent)
                        existing.intents = merged_intents
                        st.session_state.intent_set = existing
                    else:
                        st.session_state.intent_set = result

                    st.success(f"Generated {len(st.session_state.intent_set.intents)} intents")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        # Upload existing
        st.divider()
        uploaded = st.file_uploader("Or upload existing intents JSON", type=["json"], key="intent_upload")
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                st.session_state.intent_set = IntentScenarioSet.model_validate(data)
                st.success(f"Loaded {len(st.session_state.intent_set.intents)} intents")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    with tab_edit:
        if st.session_state.intent_set is None:
            st.warning("Generate or load intents first.")
            return

        intent_set = st.session_state.intent_set

        # Summary view
        st.subheader(f"Agent: {intent_set.agent_name} — {len(intent_set.intents)} intents")

        for i, intent in enumerate(intent_set.intents):
            with st.expander(f"Intent: {intent.name} ({intent.intent_id}) — {len(intent.scenarios)} scenarios"):
                st.markdown(f"**Category:** {intent.category}")
                st.markdown(f"**Description:** {intent.description}")

                # Edit individual intent
                edited = json_editor(intent.model_dump(), key=f"intent_{i}")
                if edited != intent.model_dump():
                    try:
                        intent_set.intents[i] = Intent.model_validate(edited)
                        st.session_state.intent_set = intent_set
                        st.success("Intent updated.")
                    except Exception as e:
                        st.error(f"Invalid intent data: {e}")

        # Save
        st.divider()
        col_save, col_download = st.columns(2)
        with col_save:
            save_path = st.text_input("Save path", value=f"{intent_set.agent_name}_intents.json", key="intent_save_path")
            if st.button("Save to disk", key="save_intents"):
                Path(save_path).write_text(intent_set.model_dump_json(indent=2))
                st.success(f"Saved to {save_path}")
        with col_download:
            st.download_button(
                "Download JSON",
                data=intent_set.model_dump_json(indent=2),
                file_name=f"{intent_set.agent_name}_intents.json",
                mime="application/json",
            )
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/intent_manager.py
git commit -m "feat: add intent/scenario manager Streamlit page with selective regeneration"
```

---

## Task 9: Streamlit UI — Test Case Manager Page

**Files:**
- Create: `adk_eval_tool/ui/pages/testcase_manager.py`

The test case manager provides trajectory-aware editing of ADK EvalSets. Each eval case
can be either a static trajectory (conversation with toolUses, toolResponses,
intermediateResponses, rubrics) or a dynamic conversation_scenario. The UI renders
each invocation turn with structured editors for tool trajectory, expected responses,
and reference data — not just raw JSON. Users can add, edit, and delete individual
eval cases within an eval set.

- [ ] **Step 1: Implement test case manager page**

```python
# adk_eval_tool/ui/pages/testcase_manager.py
"""Page: Generate, view, and edit evaluation test cases with trajectory support."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import TestCaseConfig
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Test Cases (EvalSets)")

    if st.session_state.metadata is None:
        st.warning("Load agent metadata first.")
        return

    tab_gen, tab_edit = st.tabs(["Generate", "View / Edit"])

    with tab_gen:
        _render_generate_tab()

    with tab_edit:
        _render_edit_tab()


def _render_generate_tab():
    st.subheader("Generate Test Cases")

    if st.session_state.intent_set is None:
        st.warning("Generate or load intents first (Intents & Scenarios page).")
        return

    col1, col2 = st.columns(2)
    with col1:
        judge_model = st.selectbox("Judge model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])
        match_type = st.selectbox("Tool trajectory match type", ["IN_ORDER", "EXACT", "ANY_ORDER"])
    with col2:
        num_runs = st.number_input("Evaluation runs", min_value=1, max_value=10, value=2)
        trajectory_threshold = st.slider("Tool trajectory threshold", 0.0, 1.0, 0.8)
        safety_threshold = st.slider("Safety threshold", 0.0, 1.0, 1.0)

    config = TestCaseConfig(
        eval_metrics={
            "tool_trajectory_avg_score": trajectory_threshold,
            "safety_v1": safety_threshold,
        },
        judge_model=judge_model,
        num_runs=num_runs,
        tool_trajectory_match_type=match_type,
    )

    intent_set = st.session_state.intent_set
    intent_options = {i.intent_id: i.name for i in intent_set.intents}
    selected = st.multiselect(
        "Generate for intents (all if empty)",
        options=list(intent_options.keys()),
        format_func=lambda x: f"{x} — {intent_options[x]}",
    )

    save_dir = st.text_input("Save directory", value="eval_datasets")

    if st.button("Generate Test Cases"):
        with st.spinner("Generating evaluation datasets..."):
            from adk_eval_tool.testcase_generator import generate_test_cases

            intents_to_process = (
                [i for i in intent_set.intents if i.intent_id in selected]
                if selected
                else intent_set.intents
            )

            results = []
            progress = st.progress(0)
            for idx, intent in enumerate(intents_to_process):
                try:
                    eval_set = asyncio.run(generate_test_cases(
                        metadata=st.session_state.metadata,
                        intent=intent,
                        config=config,
                        save_dir=save_dir,
                    ))
                    results.append(eval_set)
                except Exception as e:
                    st.error(f"Failed for intent {intent.intent_id}: {e}")
                progress.progress((idx + 1) / len(intents_to_process))

            st.session_state.eval_sets = results
            st.success(f"Generated {len(results)} eval sets")

    # Upload existing evalset.json
    st.divider()
    uploaded = st.file_uploader("Or upload existing .evalset.json", type=["json"], key="evalset_upload")
    if uploaded:
        try:
            data = json.loads(uploaded.read())
            if "evalSetId" in data:
                st.session_state.eval_sets.append(data)
                st.success(f"Loaded eval set: {data.get('evalSetId')}")
            else:
                st.error("File does not appear to be an ADK EvalSet (missing evalSetId)")
        except Exception as e:
            st.error(f"Failed to load: {e}")


def _render_edit_tab():
    if not st.session_state.eval_sets:
        st.warning("Generate or upload test cases first.")
        return

    for es_idx, eval_set in enumerate(st.session_state.eval_sets):
        eval_set_id = eval_set.get("evalSetId", f"evalset_{es_idx}")
        cases = eval_set.get("evalCases", [])

        with st.expander(f"EvalSet: {eval_set_id} — {len(cases)} case(s)", expanded=True):
            # Eval set metadata
            col_name, col_desc = st.columns(2)
            with col_name:
                eval_set["name"] = st.text_input(
                    "Name", value=eval_set.get("name", ""), key=f"es_name_{es_idx}"
                )
            with col_desc:
                eval_set["description"] = st.text_input(
                    "Description", value=eval_set.get("description", ""), key=f"es_desc_{es_idx}"
                )

            # Per-case editing
            cases_to_delete = []
            for case_idx, case in enumerate(cases):
                eval_id = case.get("evalId", f"case_{case_idx}")
                is_dynamic = "conversation_scenario" in case

                with st.container():
                    col_title, col_del = st.columns([5, 1])
                    with col_title:
                        st.markdown(f"#### {'🔄' if is_dynamic else '📋'} Case: `{eval_id}`")
                    with col_del:
                        if st.button("Delete", key=f"del_case_{es_idx}_{case_idx}", type="secondary"):
                            cases_to_delete.append(case_idx)

                    if is_dynamic:
                        # Conversation scenario mode
                        _render_conversation_scenario_editor(case, es_idx, case_idx)
                    else:
                        # Trajectory mode: structured editing
                        _render_trajectory_editor(case, es_idx, case_idx)

                    st.divider()

            # Delete marked cases
            for idx in sorted(cases_to_delete, reverse=True):
                cases.pop(idx)
                st.rerun()

            # Add new case
            st.markdown("**Add new eval case:**")
            col_type, col_add = st.columns([2, 1])
            with col_type:
                new_case_type = st.radio(
                    "Type", ["Trajectory (static)", "Conversation Scenario (dynamic)"],
                    key=f"new_type_{es_idx}", horizontal=True,
                )
            with col_add:
                if st.button("Add Case", key=f"add_case_{es_idx}"):
                    new_id = f"new_case_{uuid.uuid4().hex[:6]}"
                    if "Trajectory" in new_case_type:
                        cases.append({
                            "evalId": new_id,
                            "conversation": [{
                                "invocationId": "inv-1",
                                "userContent": {"role": "user", "parts": [{"text": ""}]},
                                "intermediateData": {"toolUses": [], "toolResponses": [], "intermediateResponses": []},
                            }],
                        })
                    else:
                        cases.append({
                            "evalId": new_id,
                            "conversation_scenario": {"starting_prompt": "", "conversation_plan": ""},
                        })
                    st.rerun()

            # Raw JSON fallback
            with st.expander("Raw JSON editor"):
                edited = json_editor(eval_set, key=f"evalset_raw_{es_idx}")
                if edited != eval_set:
                    st.session_state.eval_sets[es_idx] = edited
                    st.success("EvalSet updated from raw JSON.")

    # Save all
    st.divider()
    save_dir = st.text_input("Save directory", value="eval_datasets", key="save_dir_edit")
    if st.button("Save all to disk", key="save_evalsets"):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        for eval_set in st.session_state.eval_sets:
            eval_id = eval_set.get("evalSetId", "unknown")
            (path / f"{eval_id}.evalset.json").write_text(json.dumps(eval_set, indent=2))
        st.success(f"Saved {len(st.session_state.eval_sets)} eval sets to {save_dir}/")


def _render_trajectory_editor(case: dict, es_idx: int, case_idx: int):
    """Render structured editor for trajectory-based eval cases."""
    conversation = case.get("conversation", [])

    # Session input
    session_input = case.get("sessionInput")
    if session_input:
        with st.expander("Session Input (initial state)"):
            state_str = json.dumps(session_input.get("state", {}), indent=2)
            new_state_str = st.text_area("State", value=state_str, key=f"sess_state_{es_idx}_{case_idx}")
            try:
                case["sessionInput"]["state"] = json.loads(new_state_str)
            except json.JSONDecodeError:
                pass

    for inv_idx, inv in enumerate(conversation):
        inv_id = inv.get("invocationId", f"inv-{inv_idx + 1}")

        st.markdown(f"**Turn {inv_idx + 1}** (`{inv_id}`)")

        # User message
        user_text = ""
        if inv.get("userContent", {}).get("parts"):
            user_text = inv["userContent"]["parts"][0].get("text", "")
        new_user_text = st.text_area(
            "User message",
            value=user_text,
            key=f"user_{es_idx}_{case_idx}_{inv_idx}",
        )
        inv["userContent"] = {"role": "user", "parts": [{"text": new_user_text}]}

        # Tool trajectory
        intermediate = inv.get("intermediateData", {})

        col_tools, col_responses = st.columns(2)
        with col_tools:
            st.markdown("**Expected tool calls** (toolUses)")
            tool_uses = intermediate.get("toolUses", [])
            tool_uses_str = json.dumps(tool_uses, indent=2)
            new_tool_uses_str = st.text_area(
                "toolUses",
                value=tool_uses_str,
                height=120,
                key=f"tool_uses_{es_idx}_{case_idx}_{inv_idx}",
                help='[{"name": "tool_name", "args": {"key": "value"}}]',
            )
            try:
                intermediate["toolUses"] = json.loads(new_tool_uses_str)
            except json.JSONDecodeError:
                pass

        with col_responses:
            st.markdown("**Reference tool responses** (for hallucination eval)")
            tool_responses = intermediate.get("toolResponses", [])
            tool_resp_str = json.dumps(tool_responses, indent=2)
            new_tool_resp_str = st.text_area(
                "toolResponses",
                value=tool_resp_str,
                height=120,
                key=f"tool_resp_{es_idx}_{case_idx}_{inv_idx}",
                help='[{"name": "tool", "id": "call_1", "response": {...}}]',
            )
            try:
                intermediate["toolResponses"] = json.loads(new_tool_resp_str)
            except json.JSONDecodeError:
                pass

        inv["intermediateData"] = intermediate

        # Intermediate responses (agent reasoning between tool calls)
        inter_resps = intermediate.get("intermediateResponses", [])
        if inter_resps:
            with st.expander("Intermediate responses"):
                inter_str = json.dumps(inter_resps, indent=2)
                new_inter_str = st.text_area(
                    "intermediateResponses", value=inter_str,
                    key=f"inter_resp_{es_idx}_{case_idx}_{inv_idx}",
                )
                try:
                    intermediate["intermediateResponses"] = json.loads(new_inter_str)
                except json.JSONDecodeError:
                    pass

        # Expected final response
        final_text = ""
        if inv.get("finalResponse", {}).get("parts"):
            final_text = inv["finalResponse"]["parts"][0].get("text", "")
        new_final_text = st.text_area(
            "Expected response",
            value=final_text,
            key=f"final_{es_idx}_{case_idx}_{inv_idx}",
        )
        if new_final_text:
            inv["finalResponse"] = {"role": "model", "parts": [{"text": new_final_text}]}

        # Per-invocation rubric
        rubrics = inv.get("rubrics", [])
        rubric_text = ""
        if rubrics:
            rubric_text = rubrics[0].get("rubricContent", {}).get("textProperty", "")
        new_rubric = st.text_input(
            "Quality rubric (optional)",
            value=rubric_text,
            key=f"rubric_{es_idx}_{case_idx}_{inv_idx}",
        )
        if new_rubric:
            inv["rubrics"] = [{
                "rubricId": f"rubric_inv_{inv_idx + 1}",
                "rubricContent": {"textProperty": new_rubric},
            }]
        elif rubrics and not new_rubric:
            inv.pop("rubrics", None)

    # Add invocation turn
    if st.button("Add turn", key=f"add_turn_{es_idx}_{case_idx}"):
        conversation.append({
            "invocationId": f"inv-{len(conversation) + 1}",
            "userContent": {"role": "user", "parts": [{"text": ""}]},
            "intermediateData": {"toolUses": [], "toolResponses": [], "intermediateResponses": []},
        })
        st.rerun()


def _render_conversation_scenario_editor(case: dict, es_idx: int, case_idx: int):
    """Render editor for dynamic conversation_scenario eval cases."""
    cs = case.get("conversation_scenario", {})

    cs["starting_prompt"] = st.text_area(
        "Starting prompt",
        value=cs.get("starting_prompt", ""),
        key=f"cs_prompt_{es_idx}_{case_idx}",
    )
    cs["conversation_plan"] = st.text_area(
        "Conversation plan",
        value=cs.get("conversation_plan", ""),
        key=f"cs_plan_{es_idx}_{case_idx}",
        help="Instructions for the user simulator. End with 'Signal completion when done.'",
    )

    case["conversation_scenario"] = cs
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/testcase_manager.py
git commit -m "feat: add trajectory-aware test case manager with add/edit/delete eval cases"
```

---

## Task 10: Streamlit UI — Dataset Version Management

**Files:**
- Create: `adk_eval_tool/ui/pages/dataset_versions.py`

- [ ] **Step 1: Implement dataset versions page**

```python
# adk_eval_tool/ui/pages/dataset_versions.py
"""Page: Manage versions of generated datasets."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from adk_eval_tool.ui.components.json_editor import json_editor


VERSIONS_DIR = "eval_versions"


def render():
    st.header("Dataset Versions")

    tab_create, tab_browse = st.tabs(["Create Version", "Browse Versions"])

    with tab_create:
        _render_create_version()

    with tab_browse:
        _render_browse_versions()


def _render_create_version():
    st.subheader("Create New Version")

    if not st.session_state.eval_sets and not st.session_state.intent_set:
        st.warning("No data to version. Generate metadata, intents, or test cases first.")
        return

    version_name = st.text_input(
        "Version name",
        value=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    version_notes = st.text_area("Version notes", placeholder="What changed in this version...")

    source_dir = st.text_input("Source eval_datasets directory", value="eval_datasets")

    if st.button("Create Version"):
        version_path = Path(VERSIONS_DIR) / version_name
        version_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        if st.session_state.metadata:
            (version_path / "metadata.json").write_text(
                st.session_state.metadata.model_dump_json(indent=2)
            )

        # Save intents
        if st.session_state.intent_set:
            (version_path / "intents.json").write_text(
                st.session_state.intent_set.model_dump_json(indent=2)
            )

        # Copy eval sets from source directory
        src = Path(source_dir)
        if src.exists():
            evalsets_dir = version_path / "evalsets"
            evalsets_dir.mkdir(exist_ok=True)
            for f in src.glob("*.evalset.json"):
                shutil.copy2(f, evalsets_dir / f.name)

        # Also save current in-memory eval sets
        if st.session_state.eval_sets:
            evalsets_dir = version_path / "evalsets"
            evalsets_dir.mkdir(exist_ok=True)
            for es in st.session_state.eval_sets:
                eval_id = es.get("evalSetId", "unknown")
                (evalsets_dir / f"{eval_id}.evalset.json").write_text(
                    json.dumps(es, indent=2)
                )

        # Save version manifest
        manifest = {
            "version": version_name,
            "created_at": datetime.now().isoformat(),
            "notes": version_notes,
            "agent_name": st.session_state.metadata.name if st.session_state.metadata else None,
            "num_intents": len(st.session_state.intent_set.intents) if st.session_state.intent_set else 0,
            "num_eval_sets": len(st.session_state.eval_sets),
        }
        (version_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

        st.success(f"Version '{version_name}' created at {version_path}")


def _render_browse_versions():
    versions_path = Path(VERSIONS_DIR)
    if not versions_path.exists():
        st.info("No versions created yet.")
        return

    versions = sorted(versions_path.iterdir(), reverse=True)
    if not versions:
        st.info("No versions found.")
        return

    for version_dir in versions:
        if not version_dir.is_dir():
            continue

        manifest_file = version_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
        else:
            manifest = {"version": version_dir.name}

        with st.expander(
            f"v{manifest.get('version', version_dir.name)} — "
            f"{manifest.get('created_at', 'unknown')} — "
            f"{manifest.get('notes', '')[:50]}"
        ):
            st.json(manifest)

            # List eval sets in this version
            evalsets_dir = version_dir / "evalsets"
            if evalsets_dir.exists():
                st.markdown("**Eval Sets:**")
                for f in evalsets_dir.glob("*.evalset.json"):
                    st.markdown(f"- `{f.name}`")

            col_load, col_delete = st.columns(2)
            with col_load:
                if st.button(f"Load version", key=f"load_{version_dir.name}"):
                    _load_version(version_dir)
                    st.success(f"Loaded version {version_dir.name}")
                    st.rerun()
            with col_delete:
                if st.button(f"Delete version", key=f"del_{version_dir.name}", type="secondary"):
                    shutil.rmtree(version_dir)
                    st.success(f"Deleted version {version_dir.name}")
                    st.rerun()


def _load_version(version_dir: Path):
    """Load a version's data into session state."""
    from adk_eval_tool.schemas import AgentMetadata, IntentScenarioSet

    meta_file = version_dir / "metadata.json"
    if meta_file.exists():
        st.session_state.metadata = AgentMetadata.model_validate(
            json.loads(meta_file.read_text())
        )

    intents_file = version_dir / "intents.json"
    if intents_file.exists():
        st.session_state.intent_set = IntentScenarioSet.model_validate(
            json.loads(intents_file.read_text())
        )

    evalsets_dir = version_dir / "evalsets"
    if evalsets_dir.exists():
        st.session_state.eval_sets = [
            json.loads(f.read_text())
            for f in sorted(evalsets_dir.glob("*.evalset.json"))
        ]
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/dataset_versions.py
git commit -m "feat: add dataset version management Streamlit page"
```

---

## Task 11: Integration Test & Final Wiring

**Files:**
- Create: `tests/test_integration.py`
- Modify: `adk_eval_tool/__init__.py` — add top-level exports

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
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
    """Test the full pipeline: parse agent → build intents → generate evalset."""
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
```

- [ ] **Step 2: Update top-level __init__.py**

```python
# adk_eval_tool/__init__.py
"""ADK Evaluation & Testing Tool.

Introspect ADK agents, generate evaluation intents/scenarios/test cases,
and manage golden datasets via Python API or Streamlit UI.
"""

from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    Intent,
    Scenario,
    ScenarioStep,
    IntentScenarioSet,
    TestCaseConfig,
    EvalRunConfig,
    MetricConfig,
    EvalRunResult,
    TraceSpanNode,
)
from adk_eval_tool.agent_parser import parse_agent, parse_agent_async

__all__ = [
    "AgentMetadata",
    "ToolMetadata",
    "Intent",
    "Scenario",
    "ScenarioStep",
    "IntentScenarioSet",
    "TestCaseConfig",
    "EvalRunConfig",
    "MetricConfig",
    "EvalRunResult",
    "TraceSpanNode",
    "parse_agent",
    "parse_agent_async",
]
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add adk_eval_tool/__init__.py tests/test_integration.py
git commit -m "feat: add integration tests and top-level package exports"
```

---

---

## Task 12: Eval Runner — Trace Collection & Result Capture

**Files:**
- Create: `adk_eval_tool/eval_runner/__init__.py`
- Create: `adk_eval_tool/eval_runner/trace_collector.py`
- Create: `adk_eval_tool/eval_runner/result_store.py`
- Create: `adk_eval_tool/eval_runner/runner.py`
- Create: `tests/test_eval_runner.py`
- Create: `tests/test_result_store.py`

This task builds the core eval execution engine. It wraps ADK's `LocalEvalService` (the same internals `AgentEvaluator` uses) but captures the `EvalCaseResult` objects instead of asserting. It also sets up OpenTelemetry with `SqliteSpanExporter` to collect execution traces during evaluation runs.

**Key ADK internals used:**
- `google.adk.evaluation.local_eval_service.LocalEvalService` — performs inference + metric evaluation
- `google.adk.evaluation.base_eval_service.InferenceRequest`, `InferenceConfig`, `EvaluateRequest`, `EvaluateConfig`
- `google.adk.evaluation.eval_result.EvalCaseResult` — contains `overall_eval_metric_results` and `eval_metric_result_per_invocation`
- `google.adk.evaluation.eval_metrics.EvalMetricResultPerInvocation` — per-turn results with `actual_invocation` and `expected_invocation`
- `google.adk.evaluation.eval_metrics.EvalMetricResult` — individual metric score, status, threshold
- `google.adk.evaluation.eval_metrics.EvalStatus` — enum: PASSED=1, FAILED=2, NOT_EVALUATED=3
- `google.adk.telemetry.sqlite_span_exporter.SqliteSpanExporter` — persists spans to SQLite with `get_all_spans_for_session(session_id)`
- `google.adk.telemetry.setup.maybe_set_otel_providers`, `OTelHooks` — configure OTel providers with custom span processors
- Span attributes: `gcp.vertex.agent.session_id`, `gcp.vertex.agent.invocation_id`, `gcp.vertex.agent.event_id`
- Span names: `invocation`, `call_llm`, `execute_tool`, `send_data`

- [ ] **Step 1: Write trace collector tests**

```python
# tests/test_eval_runner.py
"""Tests for eval runner trace collection and result capture."""

import pytest
from unittest.mock import MagicMock, patch
from adk_eval_tool.eval_runner.trace_collector import (
    build_trace_tree,
    compute_basic_metrics,
    SpanData,
)
from adk_eval_tool.schemas import TraceSpanNode, BasicMetrics


def _make_span(span_id, name, parent_span_id=None, session_id="sess-1",
               start_time=1000, end_time=1001, attributes=None):
    """Create a mock span-like object."""
    return SpanData(
        span_id=span_id,
        name=name,
        parent_span_id=parent_span_id,
        start_time=start_time,
        end_time=end_time,
        session_id=session_id,
        attributes=attributes or {},
    )


def test_build_trace_tree_single_root():
    spans = [_make_span("s1", "invocation")]
    tree = build_trace_tree(spans)
    assert len(tree) == 1
    assert tree[0].name == "invocation"
    assert tree[0].span_id == "s1"


def test_build_trace_tree_nested():
    spans = [
        _make_span("s1", "invocation"),
        _make_span("s2", "call_llm", parent_span_id="s1", start_time=1000, end_time=1001),
        _make_span("s3", "execute_tool:search", parent_span_id="s1", start_time=1001, end_time=1002),
        _make_span("s4", "call_llm", parent_span_id="s1", start_time=1002, end_time=1003),
    ]
    tree = build_trace_tree(spans)
    assert len(tree) == 1
    root = tree[0]
    assert root.name == "invocation"
    assert len(root.children) == 3
    assert root.children[0].name == "call_llm"
    assert root.children[1].name == "execute_tool:search"


def test_build_trace_tree_deep_nesting():
    spans = [
        _make_span("s1", "invocation"),
        _make_span("s2", "call_llm", parent_span_id="s1"),
        _make_span("s3", "execute_tool:search", parent_span_id="s2"),
    ]
    tree = build_trace_tree(spans)
    root = tree[0]
    assert len(root.children) == 1
    llm = root.children[0]
    assert llm.name == "call_llm"
    assert len(llm.children) == 1
    assert llm.children[0].name == "execute_tool:search"


def test_build_trace_tree_empty():
    tree = build_trace_tree([])
    assert tree == []


def test_compute_basic_metrics():
    spans = [
        _make_span("s1", "invocation", start_time=0, end_time=5,
                    attributes={}),
        _make_span("s2", "call_llm", parent_span_id="s1", start_time=0, end_time=2,
                    attributes={"gen_ai.usage.input_tokens": "500", "gen_ai.usage.output_tokens": "100"}),
        _make_span("s3", "execute_tool:search", parent_span_id="s1", start_time=2, end_time=3),
        _make_span("s4", "call_llm", parent_span_id="s1", start_time=3, end_time=5,
                    attributes={"gen_ai.usage.input_tokens": "800", "gen_ai.usage.output_tokens": "150"}),
    ]
    tree = build_trace_tree(spans)
    metrics = compute_basic_metrics(tree[0])
    assert isinstance(metrics, BasicMetrics)
    assert metrics.num_llm_calls == 2
    assert metrics.num_tool_calls == 1
    assert metrics.total_input_tokens == 1300
    assert metrics.total_output_tokens == 250
    assert metrics.total_tokens == 1550
    assert metrics.max_context_size == 800
    assert metrics.total_duration_ms == 5000.0
```

- [ ] **Step 2: Write result store tests**

```python
# tests/test_result_store.py
"""Tests for eval result storage."""

import json
from pathlib import Path
from adk_eval_tool.eval_runner.result_store import ResultStore
from adk_eval_tool.schemas import EvalRunResult


def test_result_store_save_and_load(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    result = EvalRunResult(
        run_id="run-1",
        eval_set_id="agent__book_flight",
        eval_id="book_flight__happy_path",
        status="PASSED",
        overall_scores={"tool_trajectory_avg_score": 0.95, "safety_v1": 1.0},
        per_invocation_scores=[
            {"invocation_id": "inv-1", "scores": {"tool_trajectory_avg_score": 0.95}},
        ],
        session_id="sess-1",
        timestamp=1000.0,
    )
    store.save_result(result)

    loaded = store.load_results()
    assert len(loaded) == 1
    assert loaded[0].run_id == "run-1"
    assert loaded[0].status == "PASSED"


def test_result_store_multiple_results(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    for i in range(3):
        store.save_result(EvalRunResult(
            run_id=f"run-{i}",
            eval_set_id="agent__intent",
            eval_id=f"intent__scenario_{i}",
            status="PASSED" if i < 2 else "FAILED",
            overall_scores={"safety_v1": 1.0 if i < 2 else 0.5},
        ))

    loaded = store.load_results()
    assert len(loaded) == 3
    assert sum(1 for r in loaded if r.status == "PASSED") == 2


def test_result_store_load_by_eval_set(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    store.save_result(EvalRunResult(
        run_id="run-1", eval_set_id="set_a", eval_id="case_1", status="PASSED",
    ))
    store.save_result(EvalRunResult(
        run_id="run-2", eval_set_id="set_b", eval_id="case_2", status="FAILED",
    ))

    results_a = store.load_results(eval_set_id="set_a")
    assert len(results_a) == 1
    assert results_a[0].eval_set_id == "set_a"


def test_result_store_compute_averages(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    for score in [0.8, 0.9, 1.0]:
        store.save_result(EvalRunResult(
            run_id=f"run-{score}",
            eval_set_id="set_a",
            eval_id="case_1",
            status="PASSED",
            overall_scores={"tool_trajectory_avg_score": score, "safety_v1": 1.0},
        ))

    averages = store.compute_averages(eval_set_id="set_a")
    assert abs(averages["tool_trajectory_avg_score"] - 0.9) < 0.01
    assert averages["safety_v1"] == 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_eval_runner.py tests/test_result_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement trace_collector.py**

```python
# adk_eval_tool/eval_runner/trace_collector.py
"""OpenTelemetry trace collection for ADK evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from adk_eval_tool.schemas import TraceSpanNode


@dataclass
class SpanData:
    """Lightweight span representation extracted from OTel ReadableSpan."""

    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    session_id: str = ""
    invocation_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


def extract_span_data_from_readable(span) -> SpanData:
    """Convert an OTel ReadableSpan to SpanData.

    Args:
        span: An opentelemetry.sdk.trace.ReadableSpan instance.

    Returns:
        SpanData with extracted fields.
    """
    attrs = dict(span.attributes) if span.attributes else {}
    return SpanData(
        span_id=format(span.context.span_id, "016x"),
        name=span.name,
        parent_span_id=(
            format(span.parent.span_id, "016x") if span.parent else None
        ),
        start_time=span.start_time / 1e9 if span.start_time else 0.0,
        end_time=span.end_time / 1e9 if span.end_time else 0.0,
        session_id=attrs.get("gcp.vertex.agent.session_id", ""),
        invocation_id=attrs.get("gcp.vertex.agent.invocation_id", ""),
        attributes=attrs,
    )


def build_trace_tree(spans: list[SpanData]) -> list[TraceSpanNode]:
    """Build a tree of TraceSpanNode from flat span list.

    Args:
        spans: List of SpanData (flat, with parent_span_id references).

    Returns:
        List of root TraceSpanNode (usually one per invocation).
    """
    if not spans:
        return []

    nodes: dict[str, TraceSpanNode] = {}
    for span in spans:
        nodes[span.span_id] = TraceSpanNode(
            span_id=span.span_id,
            name=span.name,
            start_time=span.start_time,
            end_time=span.end_time,
            attributes=span.attributes,
        )

    roots = []
    for span in spans:
        node = nodes[span.span_id]
        if span.parent_span_id and span.parent_span_id in nodes:
            nodes[span.parent_span_id].children.append(node)
        else:
            roots.append(node)

    return roots


def setup_trace_collection(db_path: str):
    """Set up OTel with SqliteSpanExporter for trace persistence.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        The SqliteSpanExporter instance (for later span retrieval).
    """
    from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter
    from google.adk.telemetry.setup import maybe_set_otel_providers, OTelHooks
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = SqliteSpanExporter(db_path=db_path)
    hooks = OTelHooks(span_processors=[BatchSpanProcessor(exporter)])
    maybe_set_otel_providers(otel_hooks_to_setup=[hooks])
    return exporter


def get_trace_tree_for_session(exporter, session_id: str) -> list[TraceSpanNode]:
    """Retrieve and build trace tree for a session from SqliteSpanExporter.

    Args:
        exporter: SqliteSpanExporter instance.
        session_id: The session ID to retrieve spans for.

    Returns:
        List of root TraceSpanNode for the session.
    """
    raw_spans = exporter.get_all_spans_for_session(session_id)
    span_data = [extract_span_data_from_readable(s) for s in raw_spans]
    return build_trace_tree(span_data)


def compute_basic_metrics(trace_tree: TraceSpanNode) -> "BasicMetrics":
    """Compute standard metrics from a trace tree.

    Walks the trace tree to extract token counts, call counts, durations,
    and response sizes from span attributes.

    Args:
        trace_tree: Root TraceSpanNode.

    Returns:
        BasicMetrics with aggregated values.
    """
    from adk_eval_tool.schemas import BasicMetrics

    total_input_tokens = 0
    total_output_tokens = 0
    num_llm_calls = 0
    num_tool_calls = 0
    max_context_size = 0
    response_lengths: list[int] = []

    def _walk(node: TraceSpanNode):
        nonlocal total_input_tokens, total_output_tokens, num_llm_calls
        nonlocal num_tool_calls, max_context_size

        if "call_llm" in node.name:
            num_llm_calls += 1
            inp = int(node.attributes.get("gen_ai.usage.input_tokens", 0))
            out = int(node.attributes.get("gen_ai.usage.output_tokens", 0))
            total_input_tokens += inp
            total_output_tokens += out
            if inp > max_context_size:
                max_context_size = inp
        elif "execute_tool" in node.name:
            num_tool_calls += 1

        for child in node.children:
            _walk(child)

    _walk(trace_tree)

    total_duration_ms = (trace_tree.end_time - trace_tree.start_time) * 1000

    return BasicMetrics(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_input_tokens + total_output_tokens,
        num_llm_calls=num_llm_calls,
        num_tool_calls=num_tool_calls,
        total_duration_ms=max(0, total_duration_ms),
        max_context_size=max_context_size,
    )
```

- [ ] **Step 5: Implement result_store.py**

```python
# adk_eval_tool/eval_runner/result_store.py
"""Persist and query evaluation results."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

from adk_eval_tool.schemas import EvalRunResult


class ResultStore:
    """File-based store for evaluation results."""

    def __init__(self, base_dir: str = "eval_results"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: EvalRunResult) -> Path:
        """Save an eval run result to disk.

        Args:
            result: The evaluation result to save.

        Returns:
            Path to the saved file.
        """
        filename = f"{result.run_id}.json"
        path = self._base_dir / filename
        path.write_text(result.model_dump_json(indent=2))
        return path

    def load_results(
        self,
        eval_set_id: Optional[str] = None,
    ) -> list[EvalRunResult]:
        """Load all results, optionally filtered by eval_set_id.

        Args:
            eval_set_id: If provided, only return results for this eval set.

        Returns:
            List of EvalRunResult.
        """
        results = []
        for path in sorted(self._base_dir.glob("*.json")):
            try:
                result = EvalRunResult.model_validate_json(path.read_text())
                if eval_set_id is None or result.eval_set_id == eval_set_id:
                    results.append(result)
            except Exception:
                continue
        return results

    def compute_averages(
        self,
        eval_set_id: Optional[str] = None,
    ) -> dict[str, float]:
        """Compute average scores across all results for an eval set.

        Args:
            eval_set_id: Filter by eval set.

        Returns:
            Dict of metric_name → average score.
        """
        results = self.load_results(eval_set_id=eval_set_id)
        if not results:
            return {}

        all_scores: dict[str, list[float]] = {}
        for result in results:
            for metric, score in result.overall_scores.items():
                if score is not None:
                    all_scores.setdefault(metric, []).append(score)

        return {
            metric: statistics.mean(scores)
            for metric, scores in all_scores.items()
        }
```

- [ ] **Step 6: Implement runner.py**

```python
# adk_eval_tool/eval_runner/__init__.py
"""Evaluation runner with trace collection and result capture."""

from adk_eval_tool.eval_runner.runner import run_evaluation
from adk_eval_tool.eval_runner.result_store import ResultStore

__all__ = ["run_evaluation", "ResultStore"]
```

```python
# adk_eval_tool/eval_runner/runner.py
"""Run ADK evaluations with trace collection and structured result capture.

Uses ADK's LocalEvalService internally (same as AgentEvaluator) but
captures EvalCaseResult objects instead of asserting, and sets up
OpenTelemetry tracing to record execution spans.
"""

from __future__ import annotations

import importlib
import uuid
from typing import AsyncGenerator, Optional

from google.adk.agents.base_agent import BaseAgent

from adk_eval_tool.schemas import (
    EvalRunConfig,
    EvalRunResult,
    MetricConfig,
)
from adk_eval_tool.eval_runner.trace_collector import (
    setup_trace_collection,
    get_trace_tree_for_session,
)
from adk_eval_tool.eval_runner.result_store import ResultStore


def _build_eval_config_from_metrics(metrics: list[MetricConfig], judge_model: str):
    """Convert our MetricConfig list to ADK EvalConfig.

    Maps MetricConfig → ADK criterion types:
    - tool_trajectory_avg_score → ToolTrajectoryCriterion(threshold, match_type)
    - hallucinations_v1 → HallucinationsCriterion(threshold, judge_model_options, evaluate_intermediate)
    - rubric_based_* → RubricsBasedCriterion(threshold, rubric, judge_model_options)
    - final_response_match_v2 → LlmAsAJudgeCriterion(threshold, judge_model_options)
    - Others → BaseCriterion(threshold)
    """
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.eval_metrics import (
        BaseCriterion,
        ToolTrajectoryCriterion,
        HallucinationsCriterion,
        LlmAsAJudgeCriterion,
        JudgeModelOptions,
    )

    criteria: dict = {}
    for mc in metrics:
        jm = mc.judge_model or judge_model
        judge_opts = JudgeModelOptions(judge_model=jm, num_samples=mc.judge_num_samples)

        if mc.metric_name == "tool_trajectory_avg_score":
            # MatchType enum: EXACT=0, IN_ORDER=1, ANY_ORDER=2
            from google.adk.evaluation.eval_metrics import ToolTrajectoryCriterion
            match_type_val = {"EXACT": 0, "IN_ORDER": 1, "ANY_ORDER": 2}.get(
                mc.match_type or "IN_ORDER", 1
            )
            # Get the MatchType enum from ToolTrajectoryCriterion's field
            mt_field = ToolTrajectoryCriterion.model_fields["match_type"]
            mt_enum = mt_field.annotation
            mt = mt_enum(match_type_val)
            criteria[mc.metric_name] = ToolTrajectoryCriterion(
                threshold=mc.threshold, match_type=mt
            )
        elif mc.metric_name == "hallucinations_v1":
            criteria[mc.metric_name] = HallucinationsCriterion(
                threshold=mc.threshold,
                judge_model_options=judge_opts,
                evaluate_intermediate_nl_responses=mc.evaluate_intermediate,
            )
        elif mc.metric_name in (
            "final_response_match_v2",
            "response_evaluation_score",
        ):
            criteria[mc.metric_name] = LlmAsAJudgeCriterion(
                threshold=mc.threshold,
                judge_model_options=judge_opts,
            )
        else:
            criteria[mc.metric_name] = BaseCriterion(threshold=mc.threshold)

    return EvalConfig(criteria=criteria)


def _get_agent(module_name: str, agent_name: Optional[str] = None) -> BaseAgent:
    """Load agent from module."""
    module = importlib.import_module(module_name)
    agent_module = module.agent if hasattr(module, "agent") else module
    if hasattr(agent_module, "root_agent"):
        root = agent_module.root_agent
    else:
        raise ValueError(f"Module {module_name} has no root_agent")

    if agent_name:
        found = root.find_agent(agent_name)
        if not found:
            raise ValueError(f"Agent '{agent_name}' not found")
        return found
    return root


async def run_evaluation(
    config: EvalRunConfig,
    eval_sets: list[dict],
    result_store: Optional[ResultStore] = None,
) -> list[EvalRunResult]:
    """Run ADK evaluation with trace collection and result capture.

    This mirrors what AgentEvaluator._get_eval_results_by_eval_id does
    internally, but captures EvalCaseResult objects and converts them to
    our EvalRunResult format, plus collects OTel traces.

    Args:
        config: Evaluation run configuration.
        eval_sets: List of EvalSet dicts (camelCase, ADK format).
        result_store: Optional ResultStore to persist results.

    Returns:
        List of EvalRunResult with scores and trace trees.
    """
    from google.adk.evaluation.eval_set import EvalSet
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.base_eval_service import (
        InferenceRequest,
        InferenceConfig,
        EvaluateRequest,
        EvaluateConfig,
    )
    from google.adk.evaluation.local_eval_service import LocalEvalService
    from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from google.adk.evaluation.eval_metrics import EvalStatus
    from google.adk.evaluation.agent_evaluator import AgentEvaluator
    from contextlib import aclosing as Aclosing
    import json
    import time

    # Set up trace collection
    exporter = setup_trace_collection(config.trace_db_path)

    # Load agent
    agent = _get_agent(config.agent_module, config.agent_name)

    # Build ADK eval config from our metric configs
    eval_config = _build_eval_config_from_metrics(config.metrics, config.judge_model)
    from google.adk.evaluation.eval_config import get_eval_metrics_from_config
    eval_metrics = get_eval_metrics_from_config(eval_config)

    all_results: list[EvalRunResult] = []
    app_name = "eval_app"

    for eval_set_dict in eval_sets:
        eval_set = EvalSet.model_validate(eval_set_dict)

        # Set up eval sets manager
        eval_sets_manager = InMemoryEvalSetsManager()
        eval_sets_manager.create_eval_set(
            app_name=app_name, eval_set_id=eval_set.eval_set_id
        )
        for eval_case in eval_set.eval_cases:
            eval_sets_manager.add_eval_case(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                eval_case=eval_case,
            )

        eval_service = LocalEvalService(
            root_agent=agent,
            eval_sets_manager=eval_sets_manager,
        )

        # Run inference
        inference_requests = [
            InferenceRequest(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                inference_config=InferenceConfig(),
            )
        ] * config.num_runs

        inference_results = []
        for req in inference_requests:
            async with Aclosing(
                eval_service.perform_inference(inference_request=req)
            ) as agen:
                async for result in agen:
                    inference_results.append(result)

        # Run evaluation
        evaluate_request = EvaluateRequest(
            inference_results=inference_results,
            evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
        )
        async with Aclosing(
            eval_service.evaluate(evaluate_request=evaluate_request)
        ) as agen:
            async for eval_case_result in agen:
                run_id = f"run-{uuid.uuid4().hex[:8]}"

                # Extract overall scores
                overall_scores = {}
                for metric_result in eval_case_result.overall_eval_metric_results:
                    overall_scores[metric_result.metric_name] = metric_result.score

                # Extract per-invocation scores
                per_inv_scores = []
                for per_inv in eval_case_result.eval_metric_result_per_invocation:
                    # Extract tool call details from intermediate data
                    actual_tools = []
                    expected_tools = []
                    if per_inv.actual_invocation.intermediate_data:
                        idata = per_inv.actual_invocation.intermediate_data
                        if hasattr(idata, "tool_uses"):
                            actual_tools = [
                                {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                                for tc in idata.tool_uses
                            ]
                    if per_inv.expected_invocation and per_inv.expected_invocation.intermediate_data:
                        edata = per_inv.expected_invocation.intermediate_data
                        if hasattr(edata, "tool_uses"):
                            expected_tools = [
                                {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                                for tc in edata.tool_uses
                            ]

                    inv_data = {
                        "invocation_id": per_inv.actual_invocation.invocation_id,
                        "user_message": _content_to_text(per_inv.actual_invocation.user_content),
                        "actual_response": _content_to_text(per_inv.actual_invocation.final_response),
                        "expected_response": _content_to_text(
                            per_inv.expected_invocation.final_response
                        ) if per_inv.expected_invocation else "",
                        "actual_tool_calls": actual_tools,
                        "expected_tool_calls": expected_tools,
                        "scores": {
                            mr.metric_name: mr.score
                            for mr in per_inv.eval_metric_results
                        },
                    }
                    per_inv_scores.append(inv_data)

                # Get trace tree for this session
                trace_tree = None
                if eval_case_result.session_id:
                    try:
                        exporter.force_flush()
                        trees = get_trace_tree_for_session(
                            exporter, eval_case_result.session_id
                        )
                        trace_tree = trees[0] if trees else None
                    except Exception:
                        pass

                status_map = {
                    EvalStatus.PASSED: "PASSED",
                    EvalStatus.FAILED: "FAILED",
                    EvalStatus.NOT_EVALUATED: "NOT_EVALUATED",
                }

                # Compute basic metrics from trace tree
                basic_metrics = None
                if trace_tree:
                    from adk_eval_tool.eval_runner.trace_collector import compute_basic_metrics
                    basic_metrics = compute_basic_metrics(trace_tree)

                run_result = EvalRunResult(
                    run_id=run_id,
                    eval_set_id=eval_set.eval_set_id,
                    eval_id=eval_case_result.eval_id,
                    status=status_map.get(
                        eval_case_result.final_eval_status, "NOT_EVALUATED"
                    ),
                    overall_scores=overall_scores,
                    per_invocation_scores=per_inv_scores,
                    basic_metrics=basic_metrics,
                    session_id=eval_case_result.session_id,
                    trace_tree=trace_tree,
                    timestamp=time.time(),
                )

                all_results.append(run_result)
                if result_store:
                    result_store.save_result(run_result)

    return all_results


def _content_to_text(content) -> str:
    """Extract text from genai Content object."""
    if content and content.parts:
        return "\n".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
    return ""
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_eval_runner.py tests/test_result_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 8: Commit**

```bash
git add adk_eval_tool/eval_runner/ tests/test_eval_runner.py tests/test_result_store.py
git commit -m "feat: add eval runner with OTel trace collection and result capture"
```

---

## Task 13: Streamlit UI — Eval Config Page

**Files:**
- Create: `adk_eval_tool/ui/pages/eval_config.py`

This page lets users configure evaluation metrics and thresholds using all 9 ADK built-in metrics. Each metric has its own settings (threshold, match type, judge model, etc.).

- [ ] **Step 1: Implement eval config page**

```python
# adk_eval_tool/ui/pages/eval_config.py
"""Page: Configure evaluation metrics, thresholds, and judge model."""

from __future__ import annotations

import streamlit as st

from adk_eval_tool.schemas import EvalRunConfig, MetricConfig


# All 9 ADK PrebuiltMetrics with descriptions and config options
BUILTIN_METRICS = {
    "tool_trajectory_avg_score": {
        "description": "Tool call sequence matches expected trajectory",
        "has_match_type": True,
        "default_threshold": 0.8,
    },
    "response_match_score": {
        "description": "ROUGE-1 unigram overlap with reference answer",
        "has_match_type": False,
        "default_threshold": 0.8,
    },
    "response_evaluation_score": {
        "description": "General response quality score (LLM judge)",
        "has_match_type": False,
        "default_threshold": 0.7,
    },
    "final_response_match_v2": {
        "description": "LLM-judged semantic equivalence to reference",
        "has_match_type": False,
        "default_threshold": 0.8,
    },
    "rubric_based_final_response_quality_v1": {
        "description": "LLM-judged quality against custom rubric",
        "has_match_type": False,
        "has_rubric": True,
        "default_threshold": 0.8,
    },
    "rubric_based_tool_use_quality_v1": {
        "description": "LLM-judged tool usage quality against rubric",
        "has_match_type": False,
        "has_rubric": True,
        "default_threshold": 0.8,
    },
    "hallucinations_v1": {
        "description": "Response grounded in tool outputs (no hallucinations)",
        "has_match_type": False,
        "has_evaluate_intermediate": True,
        "default_threshold": 0.9,
    },
    "safety_v1": {
        "description": "Response is safe and harmless",
        "has_match_type": False,
        "default_threshold": 1.0,
    },
    "per_turn_user_simulator_quality_v1": {
        "description": "User simulator fidelity (for conversation_scenario tests only)",
        "has_match_type": False,
        "default_threshold": 0.8,
    },
}


def render():
    st.header("Evaluation Configuration")

    st.subheader("Agent Module")
    col1, col2 = st.columns(2)
    with col1:
        agent_module = st.text_input(
            "Agent module path",
            value=st.session_state.get("_eval_agent_module", ""),
            placeholder="my_agent.agent",
            key="eval_agent_module",
        )
    with col2:
        agent_name = st.text_input(
            "Agent name (empty = root_agent)",
            value=st.session_state.get("_eval_agent_name", ""),
            key="eval_agent_name",
        )

    st.subheader("Judge Model & Runs")
    col1, col2, col3 = st.columns(3)
    with col1:
        judge_model = st.selectbox(
            "Judge model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            key="eval_judge_model",
        )
    with col2:
        num_runs = st.number_input("Number of runs", min_value=1, max_value=10, value=2, key="eval_num_runs")
    with col3:
        trace_db = st.text_input("Trace DB path", value="eval_traces.db", key="eval_trace_db")

    st.divider()
    st.subheader("Metrics")
    st.markdown("Enable metrics and configure thresholds. All 9 ADK built-in metrics are available.")

    metrics: list[MetricConfig] = []
    for metric_name, info in BUILTIN_METRICS.items():
        col_enable, col_config = st.columns([1, 3])
        with col_enable:
            enabled = st.checkbox(
                metric_name,
                value=metric_name in ("tool_trajectory_avg_score", "safety_v1"),
                key=f"enable_{metric_name}",
            )
        if not enabled:
            continue

        with col_config:
            st.caption(info["description"])
            subcols = st.columns(3)

            with subcols[0]:
                threshold = st.slider(
                    "Threshold",
                    0.0, 1.0,
                    value=info["default_threshold"],
                    key=f"thresh_{metric_name}",
                )

            match_type = None
            if info.get("has_match_type"):
                with subcols[1]:
                    match_type = st.selectbox(
                        "Match type",
                        ["IN_ORDER", "EXACT", "ANY_ORDER"],
                        key=f"match_{metric_name}",
                    )

            evaluate_intermediate = False
            if info.get("has_evaluate_intermediate"):
                with subcols[1]:
                    evaluate_intermediate = st.checkbox(
                        "Evaluate intermediate responses",
                        key=f"intermediate_{metric_name}",
                    )

            rubric = None
            if info.get("has_rubric"):
                rubric = st.text_area(
                    "Rubric",
                    placeholder="Define quality criteria...",
                    key=f"rubric_{metric_name}",
                )

            metrics.append(MetricConfig(
                metric_name=metric_name,
                threshold=threshold,
                match_type=match_type,
                evaluate_intermediate=evaluate_intermediate,
                rubric=rubric if rubric else None,
            ))

    st.divider()

    if st.button("Save Configuration", type="primary"):
        eval_config = EvalRunConfig(
            agent_module=agent_module,
            agent_name=agent_name or None,
            metrics=metrics,
            judge_model=judge_model,
            num_runs=num_runs,
            trace_db_path=trace_db,
        )
        st.session_state.eval_run_config = eval_config
        st.session_state._eval_agent_module = agent_module
        st.session_state._eval_agent_name = agent_name
        st.success(f"Saved config: {len(metrics)} metrics enabled")

    # Show current config summary
    if st.session_state.eval_run_config:
        config = st.session_state.eval_run_config
        st.subheader("Current Config")
        st.json(config.model_dump())
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/eval_config.py
git commit -m "feat: add eval config page with all 9 ADK built-in metrics"
```

---

## Task 14: Streamlit UI — Eval Launcher Page

**Files:**
- Create: `adk_eval_tool/ui/pages/eval_launcher.py`

- [ ] **Step 1: Implement eval launcher page**

```python
# adk_eval_tool/ui/pages/eval_launcher.py
"""Page: Launch evaluation runs and monitor progress."""

from __future__ import annotations

import asyncio
import json

import streamlit as st

from adk_eval_tool.eval_runner import run_evaluation, ResultStore
from adk_eval_tool.ui.components.run_status import run_status_display


def render():
    st.header("Run Evaluation")

    if st.session_state.eval_run_config is None:
        st.warning("Configure evaluation first (Eval Config page).")
        return
    if not st.session_state.eval_sets:
        st.warning("Generate test cases first (Test Cases page).")
        return

    config = st.session_state.eval_run_config

    # Show config summary
    st.subheader("Run Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agent", config.agent_module)
    with col2:
        st.metric("Metrics", len(config.metrics))
    with col3:
        st.metric("Runs per case", config.num_runs)

    # Select eval sets to run
    eval_set_options = {
        es.get("evalSetId", f"set_{i}"): es.get("name", es.get("evalSetId", f"set_{i}"))
        for i, es in enumerate(st.session_state.eval_sets)
    }
    selected_ids = st.multiselect(
        "Run for eval sets (all if empty)",
        options=list(eval_set_options.keys()),
        format_func=lambda x: eval_set_options[x],
    )

    sets_to_run = (
        [es for es in st.session_state.eval_sets if es.get("evalSetId") in selected_ids]
        if selected_ids
        else st.session_state.eval_sets
    )

    st.info(
        f"Will run **{len(sets_to_run)}** eval set(s) x "
        f"**{config.num_runs}** run(s) each. "
        f"Agent calls real tools during evaluation."
    )

    if st.button("Launch Evaluation", type="primary", disabled=not config.agent_module):
        result_store = ResultStore()
        progress = st.progress(0, text="Starting evaluation...")
        status_container = st.empty()

        try:
            results = asyncio.run(run_evaluation(
                config=config,
                eval_sets=sets_to_run,
                result_store=result_store,
            ))

            st.session_state.eval_results = results
            progress.progress(1.0, text="Evaluation complete!")

            # Summary
            passed = sum(1 for r in results if r.status == "PASSED")
            failed = sum(1 for r in results if r.status == "FAILED")

            if failed == 0:
                st.success(f"All {passed} eval cases PASSED")
            else:
                st.error(f"{failed} FAILED, {passed} PASSED out of {len(results)} cases")

            # Quick results table
            st.subheader("Results Summary")
            for result in results:
                status_icon = {"PASSED": "✅", "FAILED": "❌", "NOT_EVALUATED": "⚪"}.get(result.status, "⚪")
                scores_str = ", ".join(
                    f"{k}={v:.2f}" if v is not None else f"{k}=N/A"
                    for k, v in result.overall_scores.items()
                )
                st.markdown(f"{status_icon} **{result.eval_id}** — {scores_str}")

        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Show previous results if available
    if st.session_state.eval_results and not st.session_state.get("_eval_just_ran"):
        st.divider()
        st.subheader("Previous Run Results")
        for result in st.session_state.eval_results:
            status_icon = {"PASSED": "✅", "FAILED": "❌", "NOT_EVALUATED": "⚪"}.get(result.status, "⚪")
            st.markdown(f"{status_icon} **{result.eval_id}** — {result.status}")
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/eval_launcher.py
git commit -m "feat: add eval launcher page with progress tracking"
```

---

## Task 15: Streamlit UI — Trace Tree Component

**Files:**
- Create: `adk_eval_tool/ui/components/trace_tree.py`

- [ ] **Step 1: Implement trace tree component**

```python
# adk_eval_tool/ui/components/trace_tree.py
"""Trace tree visualization component for Streamlit."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from adk_eval_tool.schemas import TraceSpanNode


def render_trace_tree(node: TraceSpanNode, level: int = 0):
    """Render a trace span tree as nested Streamlit expanders.

    Displays span name, duration, and key attributes. Tool calls
    and LLM calls are color-coded.

    Args:
        node: Root TraceSpanNode.
        level: Current indentation level.
    """
    duration_ms = (node.end_time - node.start_time) * 1000 if node.end_time > node.start_time else 0

    # Icon and color based on span name
    if "call_llm" in node.name:
        icon = "🧠"
        label = f"LLM Call ({duration_ms:.0f}ms)"
    elif "execute_tool" in node.name:
        icon = "🔧"
        tool_name = node.name.replace("execute_tool:", "").replace("execute_tool", "tool")
        label = f"Tool: {tool_name} ({duration_ms:.0f}ms)"
    elif "invocation" in node.name:
        icon = "📋"
        label = f"Invocation ({duration_ms:.0f}ms)"
    elif "send_data" in node.name:
        icon = "📤"
        label = f"Send Data ({duration_ms:.0f}ms)"
    else:
        icon = "📎"
        label = f"{node.name} ({duration_ms:.0f}ms)"

    if node.children:
        with st.expander(f"{'  ' * level}{icon} {label}", expanded=(level < 2)):
            # Show key attributes
            _render_span_attributes(node)
            for child in node.children:
                render_trace_tree(child, level + 1)
    else:
        st.markdown(f"{'&nbsp;&nbsp;' * level * 2}{icon} {label}")
        _render_span_attributes(node)


def _render_span_attributes(node: TraceSpanNode):
    """Render relevant span attributes."""
    attrs = node.attributes
    interesting_keys = [
        "gcp.vertex.agent.invocation_id",
        "gcp.vertex.agent.event_id",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.response.finish_reasons",
    ]
    shown = {k: v for k, v in attrs.items() if k in interesting_keys}
    if shown:
        cols = st.columns(len(shown))
        for i, (key, value) in enumerate(shown.items()):
            short_key = key.split(".")[-1]
            cols[i].caption(f"{short_key}: {value}")


def render_trace_summary(nodes: list[TraceSpanNode]):
    """Render a summary of the trace tree.

    Shows counts of LLM calls, tool calls, and total duration.

    Args:
        nodes: List of root trace nodes.
    """
    if not nodes:
        st.info("No trace data available.")
        return

    llm_calls = 0
    tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def _count(node: TraceSpanNode):
        nonlocal llm_calls, tool_calls, total_input_tokens, total_output_tokens
        if "call_llm" in node.name:
            llm_calls += 1
            total_input_tokens += int(node.attributes.get("gen_ai.usage.input_tokens", 0))
            total_output_tokens += int(node.attributes.get("gen_ai.usage.output_tokens", 0))
        elif "execute_tool" in node.name:
            tool_calls += 1
        for child in node.children:
            _count(child)

    for root in nodes:
        _count(root)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Calls", llm_calls)
    col2.metric("Tool Calls", tool_calls)
    col3.metric("Input Tokens", total_input_tokens)
    col4.metric("Output Tokens", total_output_tokens)
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/components/trace_tree.py
git commit -m "feat: add trace tree visualization component"
```

---

## Task 16: Streamlit UI — Eval Results Explorer Page

**Files:**
- Create: `adk_eval_tool/ui/pages/eval_results.py`

- [ ] **Step 1: Implement eval results explorer page**

```python
# adk_eval_tool/ui/pages/eval_results.py
"""Page: Explore evaluation results, trace trees, and scores."""

from __future__ import annotations

import statistics

import streamlit as st

from adk_eval_tool.eval_runner import ResultStore
from adk_eval_tool.schemas import EvalRunResult
from adk_eval_tool.ui.components.trace_tree import render_trace_tree, render_trace_summary


def render():
    st.header("Evaluation Results Explorer")

    tab_current, tab_history = st.tabs(["Current Run", "Result History"])

    with tab_current:
        _render_current_results()

    with tab_history:
        _render_result_history()


def _render_current_results():
    results: list[EvalRunResult] = st.session_state.get("eval_results", [])
    if not results:
        st.info("No evaluation results yet. Run an evaluation first.")
        return

    # Overall summary
    st.subheader("Overall Summary")
    passed = sum(1 for r in results if r.status == "PASSED")
    failed = sum(1 for r in results if r.status == "FAILED")
    total = len(results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", total)
    col2.metric("Passed", passed, delta=f"{passed/total*100:.0f}%" if total else "0%")
    col3.metric("Failed", failed, delta=f"-{failed}" if failed else "0", delta_color="inverse")

    # Average scores across all results
    st.subheader("Average Scores")
    all_metrics: dict[str, list[float]] = {}
    for result in results:
        for metric, score in result.overall_scores.items():
            if score is not None:
                all_metrics.setdefault(metric, []).append(score)

    if all_metrics:
        metric_cols = st.columns(min(len(all_metrics), 4))
        for i, (metric, scores) in enumerate(all_metrics.items()):
            avg = statistics.mean(scores)
            metric_cols[i % len(metric_cols)].metric(
                metric,
                f"{avg:.3f}",
                delta=f"n={len(scores)}",
            )

    st.divider()

    # Per-case results with trace trees
    st.subheader("Per-Case Results")
    for result in results:
        status_icon = {"PASSED": "✅", "FAILED": "❌", "NOT_EVALUATED": "⚪"}.get(result.status, "⚪")
        with st.expander(
            f"{status_icon} {result.eval_id} — {result.status}",
            expanded=(result.status == "FAILED"),
        ):
            # Scores
            st.markdown("**Overall Scores:**")
            score_cols = st.columns(min(len(result.overall_scores), 4))
            for i, (metric, score) in enumerate(result.overall_scores.items()):
                score_str = f"{score:.3f}" if score is not None else "N/A"
                score_cols[i % len(score_cols)].metric(metric, score_str)

            # Per-invocation details
            if result.per_invocation_scores:
                st.markdown("**Per-Invocation Breakdown:**")
                for inv in result.per_invocation_scores:
                    inv_id = inv.get("invocation_id", "?")
                    user_msg = inv.get("user_message", "")
                    actual_resp = inv.get("actual_response", "")
                    expected_resp = inv.get("expected_response", "")
                    inv_scores = inv.get("scores", {})

                    with st.expander(f"Turn: {inv_id} — {user_msg[:60]}"):
                        st.markdown(f"**User:** {user_msg}")

                        # Tool trajectory comparison
                        actual_tools = inv.get("actual_tool_calls", [])
                        expected_tools = inv.get("expected_tool_calls", [])
                        if actual_tools or expected_tools:
                            col_exp_t, col_act_t = st.columns(2)
                            with col_exp_t:
                                st.markdown("**Expected Tool Calls:**")
                                for t in expected_tools:
                                    st.code(str(t), language="text")
                                if not expected_tools:
                                    st.caption("(none)")
                            with col_act_t:
                                st.markdown("**Actual Tool Calls:**")
                                for t in actual_tools:
                                    st.code(str(t), language="text")
                                if not actual_tools:
                                    st.caption("(none)")

                        # Response comparison
                        col_exp, col_act = st.columns(2)
                        with col_exp:
                            st.markdown("**Expected Response:**")
                            st.text(expected_resp or "(none)")
                        with col_act:
                            st.markdown("**Actual Response:**")
                            st.text(actual_resp or "(none)")

                        if inv_scores:
                            st.markdown("**Scores:**")
                            for metric, score in inv_scores.items():
                                score_str = f"{score:.3f}" if score is not None else "N/A"
                                st.markdown(f"- {metric}: {score_str}")

            # Basic metrics (token counts, durations, call counts)
            if result.basic_metrics:
                bm = result.basic_metrics
                st.markdown("**Basic Metrics:**")
                bm_cols = st.columns(5)
                bm_cols[0].metric("Total Tokens", f"{bm.total_tokens:,}")
                bm_cols[1].metric("Input Tokens", f"{bm.total_input_tokens:,}")
                bm_cols[2].metric("Output Tokens", f"{bm.total_output_tokens:,}")
                bm_cols[3].metric("LLM Calls", bm.num_llm_calls)
                bm_cols[4].metric("Tool Calls", bm.num_tool_calls)

                bm_cols2 = st.columns(3)
                bm_cols2[0].metric("Duration", f"{bm.total_duration_ms:.0f}ms")
                bm_cols2[1].metric("Max Context", f"{bm.max_context_size:,} tokens")
                bm_cols2[2].metric("Avg Response", f"{bm.avg_response_length:.0f} chars")

            # Trace tree
            if result.trace_tree:
                st.markdown("**Execution Trace:**")
                render_trace_summary([result.trace_tree])
                render_trace_tree(result.trace_tree)
            else:
                st.caption("No trace data available for this case.")


def _render_result_history():
    st.subheader("Historical Results")

    result_store = ResultStore()
    all_results = result_store.load_results()

    if not all_results:
        st.info("No historical results found.")
        return

    # Group by eval_set_id
    by_eval_set: dict[str, list[EvalRunResult]] = {}
    for r in all_results:
        by_eval_set.setdefault(r.eval_set_id, []).append(r)

    for eval_set_id, results in by_eval_set.items():
        with st.expander(f"EvalSet: {eval_set_id} — {len(results)} runs"):
            # Compute averages
            averages = result_store.compute_averages(eval_set_id=eval_set_id)
            if averages:
                st.markdown("**Average Scores:**")
                avg_cols = st.columns(min(len(averages), 4))
                for i, (metric, avg) in enumerate(averages.items()):
                    avg_cols[i % len(avg_cols)].metric(metric, f"{avg:.3f}")

            # Pass/fail counts
            passed = sum(1 for r in results if r.status == "PASSED")
            failed = sum(1 for r in results if r.status == "FAILED")
            st.markdown(f"**Pass rate:** {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")

            # Individual results
            for result in sorted(results, key=lambda r: r.timestamp, reverse=True):
                status_icon = {"PASSED": "✅", "FAILED": "❌"}.get(result.status, "⚪")
                scores_str = ", ".join(
                    f"{k}={v:.2f}" if v is not None else f"{k}=N/A"
                    for k, v in result.overall_scores.items()
                )
                st.markdown(f"  {status_icon} `{result.run_id}` — {result.eval_id} — {scores_str}")
```

- [ ] **Step 2: Commit**

```bash
git add adk_eval_tool/ui/pages/eval_results.py
git commit -m "feat: add eval results explorer with trace trees and score breakdowns"
```

---

## Task 17: Integration Test — Eval Runner Pipeline

**Files:**
- Modify: `tests/test_integration.py` — add eval runner integration test

- [ ] **Step 1: Add eval runner integration test**

Append to `tests/test_integration.py`:

```python
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

    # Save multiple results
    for i, (score, status) in enumerate([(0.9, "PASSED"), (0.7, "FAILED"), (0.85, "PASSED")]):
        store.save_result(EvalRunResult(
            run_id=f"run-{i}",
            eval_set_id="test_set",
            eval_id=f"case_{i}",
            status=status,
            overall_scores={"tool_trajectory_avg_score": score, "safety_v1": 1.0},
            timestamp=float(i),
        ))

    # Load all
    all_results = store.load_results()
    assert len(all_results) == 3

    # Filter by eval set
    filtered = store.load_results(eval_set_id="test_set")
    assert len(filtered) == 3

    # Averages
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
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add eval runner integration tests"
```

---

## Summary

| Task | What it delivers | Files |
|------|-----------------|-------|
| 1 | Project setup + Pydantic schemas (metadata, intents, eval types, BasicMetrics) | `pyproject.toml`, `schemas.py`, `test_schemas.py` |
| 2 | Agent parser (sync) | `agent_parser/parser.py`, `test_parser.py` |
| 3 | MCP tool resolution + async parsing | `agent_parser/mcp_resolver.py`, `test_mcp_resolver.py` |
| 4 | Intent/scenario generator ADK agent | `intent_generator/`, `test_intent_generator.py` |
| 5 | Test case generator (trajectory + conversation_scenario) | `testcase_generator/`, `test_testcase_generator.py` |
| 6 | Streamlit app shell + components | `ui/app.py`, `ui/components/` |
| 7 | Metadata viewer/editor page | `ui/pages/metadata_viewer.py` |
| 8 | Intent manager page | `ui/pages/intent_manager.py` |
| 9 | Test case manager: trajectory-aware editor with add/edit/delete cases | `ui/pages/testcase_manager.py` |
| 10 | Dataset version management page | `ui/pages/dataset_versions.py` |
| 11 | Integration tests + final wiring | `test_integration.py`, `__init__.py` |
| 12 | Eval runner: OTel traces + result capture + BasicMetrics | `eval_runner/`, `test_eval_runner.py`, `test_result_store.py` |
| 13 | Eval config page (all 9 ADK metrics with per-metric settings) | `ui/pages/eval_config.py` |
| 14 | Eval launcher page | `ui/pages/eval_launcher.py` |
| 15 | Trace tree visualization component | `ui/components/trace_tree.py` |
| 16 | Eval results explorer (traces, tool trajectory diffs, basic metrics, scores) | `ui/pages/eval_results.py` |
| 17 | Eval runner integration tests | `test_integration.py` (extended) |
