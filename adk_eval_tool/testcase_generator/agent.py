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
    Task,
    TaskScenarioSet,
    TestCaseConfig,
    TestGenConfig,
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
    gen_config: Optional[TestGenConfig] = None,
) -> Agent:
    """Create an ADK agent for test case generation."""
    gen_config = gen_config or TestGenConfig()
    return Agent(
        name="testcase_generator",
        model=gen_config.judge_model,
        description="Generates ADK-compatible evaluation test cases",
        instruction=build_testcase_system_instruction(metadata, config, gen_config),
        tools=[validate_eval_set, save_eval_set],
    )


async def generate_test_cases(
    metadata: AgentMetadata,
    task: Task,
    config: Optional[TestCaseConfig] = None,
    gen_config: Optional[TestGenConfig] = None,
    save_dir: Optional[str] = None,
) -> dict:
    """Generate an ADK EvalSet for a single task.

    Args:
        metadata: Agent metadata.
        task: The task to generate test cases for.
        config: Evaluation config (metrics, thresholds, judge model).
        save_dir: Optional directory to save the .evalset.json file.

    Returns:
        Dict in ADK EvalSet JSON format (camelCase).
    """
    config = config or TestCaseConfig()
    gen_config = gen_config or TestGenConfig()
    agent = _create_testcase_generator_agent(metadata, config, gen_config)

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="testcase_gen", session_service=session_service)

    scenario_summary = ", ".join(
        f"{sw.name}={sw.weight}%" for sw in gen_config.scenario_weights
    ) if gen_config.scenario_weights else "default distribution"
    failure_summary = ", ".join(
        f"{fw.name}={fw.weight}%" for fw in gen_config.failure_weights
    ) if gen_config.failure_weights else "default distribution"

    prompt = f"""Generate an ADK EvalSet for this task:

Task: {task.name} ({task.task_id})
Description: {task.description}

Scenarios:
{json.dumps([s.model_dump() for s in task.scenarios], indent=2)}

Generate exactly {gen_config.total_test_cases_per_task} test cases total.
Scenario distribution: {scenario_summary}
Failure distribution (within failure_path): {failure_summary}

IMPORTANT: In toolResponses, only use fields 'name', 'id', and 'response'. Do NOT include 'error' or other extra fields.

Call validate_eval_set to check your output, then call save_eval_set with the final JSON.
"""

    from adk_eval_tool.llm_runner import run_agent_with_timeout

    target_count = gen_config.total_test_cases_per_task

    session = await session_service.create_session(app_name="testcase_gen", user_id="generator")
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    events = await run_agent_with_timeout(
        runner,
        user_id="generator",
        session_id=session.id,
        new_message=content,
        timeout=180,
        max_retries=3,
        session_service=session_service,
        app_name="testcase_gen",
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
                    if fc.name == "save_eval_set" and fc.args:
                        arg_val = fc.args.get("eval_set_json", "")
                        if arg_val:
                            collected_tool_args.append(arg_val)
                    elif fc.name == "validate_eval_set" and fc.args:
                        arg_val = fc.args.get("eval_set_json", "")
                        if arg_val:
                            collected_tool_args.append(arg_val)

    # Try tool call args first (most reliable with newer models)
    eval_set = None
    for arg in reversed(collected_tool_args):
        eval_set = _extract_eval_set(arg, metadata.name, task)
        if eval_set.get("evalCases"):
            break
        eval_set = None

    # Fall back to text output
    if not eval_set:
        all_text = "\n".join(collected_texts)
        eval_set = _extract_eval_set(all_text, metadata.name, task)

    # Enforce target count: trim if too many, warn if too few
    cases = eval_set.get("evalCases", [])
    if len(cases) > target_count:
        eval_set["evalCases"] = cases[:target_count]

    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{metadata.name}__{task.task_id}.evalset.json"
        (path / filename).write_text(json.dumps(eval_set, indent=2))

    return eval_set


def _extract_eval_set(text: str, agent_name: str, task: Task) -> dict:
    """Extract EvalSet JSON from text, with fallback to programmatic build."""
    # Try parsing the whole string as JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            validation = validate_eval_set(json.dumps(data))
            if validation["valid"]:
                return data
    except (json.JSONDecodeError, Exception):
        pass

    # Try finding JSON within text
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            validation = validate_eval_set(json.dumps(data))
            if validation["valid"]:
                return data
        except (json.JSONDecodeError, Exception):
            pass

    return build_eval_set_json(task.model_dump(), agent_name)


async def generate_all_test_cases(
    metadata: AgentMetadata,
    task_set: TaskScenarioSet,
    config: Optional[TestCaseConfig] = None,
    gen_config: Optional[TestGenConfig] = None,
    save_dir: Optional[str] = None,
) -> list[dict]:
    """Generate EvalSets for all tasks in a TaskScenarioSet."""
    results = []
    for task in task_set.tasks:
        eval_set = await generate_test_cases(metadata, task, config, gen_config, save_dir)
        results.append(eval_set)
    return results
