import json
from adk_eval_tool.schemas import (
    ToolMetadata,
    AgentMetadata,
    Task,
    Trajectory,
    TrajectoryStep,
    TaskTrajectorySet,
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


def test_trajectory_step():
    step = TrajectoryStep(
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


def test_trajectory_conversation_scenario_type():
    traj = Trajectory(
        trajectory_id="dynamic_refund",
        name="Dynamic refund flow",
        eval_type="conversation_scenario",
        conversation_scenario={
            "starting_prompt": "I want a refund for order ORD-123",
            "conversation_plan": "Provide order ID when asked. Confirm refund. Signal completion.",
        },
        session_state={"user_tier": "premium"},
    )
    assert traj.eval_type == "conversation_scenario"
    assert traj.conversation_scenario is not None
    assert traj.session_state["user_tier"] == "premium"


def test_task_with_trajectories():
    traj = Trajectory(
        trajectory_id="book_flight_happy",
        name="Successful flight booking",
        description="User books a flight successfully",
        steps=[
            TrajectoryStep(
                user_message="Book a flight to London tomorrow",
                expected_tool_calls=["search_flights"],
            )
        ],
    )
    task = Task(
        task_id="book_flight",
        name="Book Flight",
        description="User wants to book a flight",
        trajectories=[traj],
    )
    assert task.task_id == "book_flight"
    assert len(task.trajectories) == 1


def test_task_trajectory_set():
    task_set = TaskTrajectorySet(
        agent_name="travel_agent",
        tasks=[
            Task(
                task_id="book_flight",
                name="Book Flight",
                description="Book a flight",
                trajectories=[],
            )
        ],
        generation_context="Testing travel agent",
    )
    assert task_set.agent_name == "travel_agent"


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
        eval_set_id="agent__task",
        eval_id="task__trajectory",
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
