"""ADK agent that generates tasks and scenarios from agent metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_eval_tool.schemas import AgentMetadata, TaskScenarioSet
from adk_eval_tool.task_generator.prompts import build_system_instruction
from adk_eval_tool.task_generator.tools import save_output, validate_task_output


def _create_task_generator_agent(metadata: AgentMetadata) -> Agent:
    """Create an ADK agent configured for task/scenario generation."""
    return Agent(
        name="task_generator",
        model="gemini-2.5-flash",
        description="Generates tasks and scenarios for an agent",
        instruction=build_system_instruction(metadata),
        tools=[save_output],
    )


async def generate_tasks(
    metadata: AgentMetadata,
    user_constraints: str = "",
    num_scenarios_per_task: int = 1,
    save_path: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> TaskScenarioSet:
    """Generate tasks and scenarios for an agent using an ADK agent.

    Args:
        metadata: Parsed agent metadata.
        user_constraints: Additional context or constraints from the user.
        num_scenarios_per_task: Target number of scenarios per task.
        save_path: Optional file path to save the output JSON.
        model: Model to use for generation.

    Returns:
        TaskScenarioSet with generated tasks and scenarios.
    """
    agent = _create_task_generator_agent(metadata)
    if model != "gemini-2.5-flash":
        agent.model = model

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="task_gen", session_service=session_service)

    user_message = f"""Generate tasks and scenarios for the agent described in your instructions.

Requirements:
- Generate {num_scenarios_per_task} scenario per task
- Cover all distinct user tasks the agent handles
- Every tool and sub-agent must appear in at least one scenario
- Each scenario should describe the input type (intent variation) and expected output type
"""
    if user_constraints:
        user_message += f"\nAdditional constraints:\n{user_constraints}"

    user_message += "\n\nCall the save_output tool with the complete JSON when done."

    from adk_eval_tool.llm_runner import run_agent_with_timeout

    session = await session_service.create_session(app_name="task_gen", user_id="generator")

    content = types.Content(role="user", parts=[types.Part(text=user_message)])

    events = await run_agent_with_timeout(
        runner,
        user_id="generator",
        session_id=session.id,
        new_message=content,
        timeout=120,
        max_retries=3,
        session_service=session_service,
        app_name="task_gen",
    )

    # Collect both text and tool call arguments from events
    collected_texts: list[str] = []
    collected_tool_args: list[str] = []

    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    collected_texts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    if fc.name == "save_output" and fc.args:
                        arg_val = fc.args.get("output_json", "")
                        if arg_val:
                            collected_tool_args.append(arg_val)

    # Try tool call args first (most reliable — the model passed JSON directly)
    task_set = None
    for arg in collected_tool_args:
        task_set = _extract_task_set(arg, metadata.name)
        if task_set.tasks:
            break

    # Fall back to text output
    if not task_set or not task_set.tasks:
        all_text = "\n".join(collected_texts)
        task_set = _extract_task_set(all_text, metadata.name)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(task_set.model_dump_json(indent=2))

    return task_set


def _extract_task_set(text: str, agent_name: str) -> TaskScenarioSet:
    """Extract TaskScenarioSet from text (JSON string or text containing JSON)."""
    # Try parsing the whole string as JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "agent_name" not in data:
                data["agent_name"] = agent_name
            return TaskScenarioSet.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Try finding JSON within text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "agent_name" not in data:
                data["agent_name"] = agent_name
            return TaskScenarioSet.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    return TaskScenarioSet(agent_name=agent_name, tasks=[], generation_context="Generation failed to produce valid JSON")
