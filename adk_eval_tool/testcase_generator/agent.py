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

    return build_eval_set_json(intent.model_dump(), agent_name)


async def generate_all_test_cases(
    metadata: AgentMetadata,
    intent_set: IntentScenarioSet,
    config: Optional[TestCaseConfig] = None,
    save_dir: Optional[str] = None,
) -> list[dict]:
    """Generate EvalSets for all intents in an IntentScenarioSet."""
    results = []
    for intent in intent_set.intents:
        eval_set = await generate_test_cases(metadata, intent, config, save_dir)
        results.append(eval_set)
    return results
