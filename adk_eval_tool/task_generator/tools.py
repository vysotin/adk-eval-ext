"""Function tools for the task/trajectory generator agent."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from adk_eval_tool.schemas import AgentMetadata, TaskTrajectorySet


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


def validate_task_output(data: dict[str, Any]) -> dict[str, Any]:
    """Validate that generated output matches TaskTrajectorySet schema.

    Args:
        data: The generated task/trajectory data as a dict.

    Returns:
        Dict with 'valid' bool and optional 'errors' list.
    """
    try:
        TaskTrajectorySet.model_validate(data)
        return {"valid": True, "errors": []}
    except ValidationError as e:
        return {
            "valid": False,
            "errors": [str(err) for err in e.errors()],
        }


def save_output(output_json: str) -> str:
    """Save the final task/trajectory output. Called by the agent when done.

    Args:
        output_json: The complete JSON string of the TaskTrajectorySet.

    Returns:
        Validation result message.
    """
    try:
        data = json.loads(output_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    validation = validate_task_output(data)
    if validation["valid"]:
        return "Output validated successfully."
    else:
        return f"Validation errors: {validation['errors']}"
