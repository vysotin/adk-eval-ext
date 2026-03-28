"""ADK agent that generates tasks and base trajectories from agent metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_eval_tool.schemas import AgentMetadata, TaskTrajectorySet
from adk_eval_tool.task_generator.prompts import build_system_instruction
from adk_eval_tool.task_generator.tools import save_output, validate_task_output


def _create_task_generator_agent(metadata: AgentMetadata) -> Agent:
    """Create an ADK agent configured for task/trajectory generation."""
    return Agent(
        name="task_generator",
        model="gemini-2.0-flash",
        description="Generates tasks and base trajectories for an agent",
        instruction=build_system_instruction(metadata),
        tools=[save_output],
    )


async def generate_tasks(
    metadata: AgentMetadata,
    user_constraints: str = "",
    num_trajectories_per_task: int = 1,
    save_path: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> TaskTrajectorySet:
    """Generate tasks and base trajectories for an agent using an ADK agent.

    Args:
        metadata: Parsed agent metadata.
        user_constraints: Additional context or constraints from the user.
        num_trajectories_per_task: Target number of base trajectories per task.
        save_path: Optional file path to save the output JSON.
        model: Model to use for generation.

    Returns:
        TaskTrajectorySet with generated tasks and trajectories.
    """
    agent = _create_task_generator_agent(metadata)
    if model != "gemini-2.0-flash":
        agent.model = model

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="task_gen", session_service=session_service)

    user_message = f"""Generate tasks and base trajectories for the agent described in your instructions.

Requirements:
- Generate {num_trajectories_per_task} base trajectory per task (happy path only)
- Cover all distinct user tasks the agent handles
- Every tool and sub-agent must appear in at least one trajectory
"""
    if user_constraints:
        user_message += f"\nAdditional constraints:\n{user_constraints}"

    user_message += "\n\nCall the save_output tool with the complete JSON when done."

    session = await session_service.create_session(app_name="task_gen", user_id="generator")

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

    task_set = _extract_task_set(final_text, metadata.name)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(task_set.model_dump_json(indent=2))

    return task_set


def _extract_task_set(text: str, agent_name: str) -> TaskTrajectorySet:
    """Extract TaskTrajectorySet from agent output text."""
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "agent_name" not in data:
                data["agent_name"] = agent_name
            return TaskTrajectorySet.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    return TaskTrajectorySet(agent_name=agent_name, tasks=[], generation_context="Generation failed to produce valid JSON")
