"""ADK Evaluation & Testing Tool.

Introspect ADK agents, generate evaluation tasks/trajectories/test cases,
and manage golden datasets via Python API or Streamlit UI.
"""

from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
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
from adk_eval_tool.agent_parser import parse_agent, parse_agent_async

__all__ = [
    "AgentMetadata",
    "ToolMetadata",
    "Task",
    "Trajectory",
    "TrajectoryStep",
    "TaskTrajectorySet",
    "TestCaseConfig",
    "EvalRunConfig",
    "MetricConfig",
    "EvalRunResult",
    "TraceSpanNode",
    "BasicMetrics",
    "parse_agent",
    "parse_agent_async",
]
