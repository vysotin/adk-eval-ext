"""Tests for Pydantic schemas."""

from adk_eval_tool.schemas import (
    ToolMetadata,
    AgentMetadata,
    Scenario,
    Task,
    TaskScenarioSet,
    MetricConfig,
    EvalRunConfig,
    TraceSpanNode,
    EvalRunResult,
    BasicMetrics,
)


def test_tool_metadata_basic():
    tool = ToolMetadata(name="search", description="Search things", parameters_schema={
        "type": "object", "properties": {"q": {"type": "string"}},
    })
    assert tool.name == "search"
    assert tool.source == "function"


def test_tool_metadata_mcp():
    tool = ToolMetadata(name="mcp_tool", description="MCP", parameters_schema={},
        source="mcp", mcp_server_name="my_server")
    assert tool.source == "mcp"


def test_agent_metadata_with_sub_agents():
    sub = AgentMetadata(name="sub", agent_type="LlmAgent")
    root = AgentMetadata(name="root", agent_type="LlmAgent", sub_agents=[sub])
    assert len(root.sub_agents) == 1


def test_agent_metadata_serialization_roundtrip():
    meta = AgentMetadata(
        name="test", agent_type="LlmAgent", description="Test agent",
        instruction="Do things", model="gemini-2.0-flash",
        tools=[ToolMetadata(name="t1", description="tool", parameters_schema={})],
    )
    loaded = AgentMetadata.model_validate_json(meta.model_dump_json())
    assert loaded.name == "test"
    assert len(loaded.tools) == 1


def test_scenario():
    sc = Scenario(
        scenario_id="happy_path", name="Happy path",
        description="User asks for weather in a known city, expects temperature and condition",
    )
    assert sc.scenario_id == "happy_path"
    assert "weather" in sc.description
    assert sc.eval_type == "scenario"


def test_scenario_conversation_scenario_type():
    sc = Scenario(
        scenario_id="dynamic_refund", name="Dynamic refund",
        description="User requests refund with valid order ID",
        eval_type="conversation_scenario",
        conversation_scenario={"starting_prompt": "Refund order 12345", "conversation_plan": "Verify and confirm."},
    )
    assert sc.eval_type == "conversation_scenario"
    assert sc.conversation_scenario is not None


def test_task_with_scenarios():
    task = Task(task_id="search", name="Search", description="Search things",
        scenarios=[Scenario(scenario_id="hp", name="Happy", description="Valid input")])
    assert len(task.scenarios) == 1


def test_task_scenario_set():
    ts = TaskScenarioSet(agent_name="agent", tasks=[
        Task(task_id="t1", name="Task", description="Desc",
            scenarios=[Scenario(scenario_id="s1", name="S", description="D")])])
    assert ts.agent_name == "agent"
    assert len(ts.tasks[0].scenarios) == 1


def test_testcase_config():
    from adk_eval_tool.schemas import TestCaseConfig
    config = TestCaseConfig()
    assert "tool_trajectory_avg_score" in config.eval_metrics


def test_metric_config():
    mc = MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.9)
    assert mc.threshold == 0.9


def test_eval_run_config():
    config = EvalRunConfig(agent_module="my.agent",
        metrics=[MetricConfig(metric_name="tool_trajectory_avg_score", threshold=0.8)])
    assert len(config.metrics) == 1


def test_trace_span_node():
    node = TraceSpanNode(span_id="s1", name="root", children=[TraceSpanNode(span_id="s2", name="child")])
    assert len(node.children) == 1


def test_eval_run_result():
    result = EvalRunResult(run_id="r1", eval_set_id="s1", eval_id="e1",
        status="PASSED", overall_scores={"metric1": 0.9})
    assert result.status == "PASSED"
