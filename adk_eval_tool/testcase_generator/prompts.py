"""Prompt templates for the test case generator agent."""

from __future__ import annotations

from typing import Optional

from adk_eval_tool.task_generator.prompts import _format_agent_tree
from adk_eval_tool.schemas import AgentMetadata, TestCaseConfig, TestGenConfig


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert at creating evaluation datasets for Google ADK agents. You generate golden test cases in the ADK `.evalset.json` format.

Given a task and its base trajectory (happy path), you generate test cases covering multiple scenario types and failure modes according to the specified distribution.

## Agent Under Test

{agent_tree}

## Evaluation Config

- Metrics: {metrics}
- Tool trajectory match type: {match_type}
- Judge model: {judge_model}

## Test Generation Parameters

- Total test cases per task: {total_per_task}

### Scenario Distribution
{scenario_distribution}

### Failure Type Distribution (within failure_path test cases)
{failure_distribution}

### Multi-Turn Configuration
{multi_turn_config}

## ADK EvalSet JSON Format

You MUST produce JSON in this exact structure (camelCase keys):

```json
{{{{
  "evalSetId": "<agent_name>__<task_id>",
  "name": "<Human-readable name>",
  "description": "<What this eval set tests>",
  "evalCases": [
    {{{{
      "evalId": "<task_id>__<scenario_type>_<short_description>",
      "conversation": [
        {{{{
          "invocationId": "inv-<n>",
          "userContent": {{{{
            "role": "user",
            "parts": [{{{{"text": "<user message>"}}}}]
          }}}},
          "finalResponse": {{{{
            "role": "model",
            "parts": [{{{{"text": "<expected response summary>"}}}}]
          }}}},
          "intermediateData": {{{{
            "toolUses": [
              {{{{"name": "<tool_name>", "args": {{{{<expected_args>}}}}}}}}
            ],
            "toolResponses": [],
            "intermediateResponses": []
          }}}}
        }}}}
      ]
    }}}}
  ]
}}}}
```

## Important Rules

1. Each conversation turn maps to one Invocation object
2. `toolUses` must list the tools the agent SHOULD call in order
3. `finalResponse` should describe what a correct response looks like
4. `evalId` format: `<task_id>__<scenario_type>_<short_description>`
5. `evalSetId` format: `<agent_name>__<task_id>`
6. Multi-turn trajectories have multiple entries in the `conversation` array
7. toolResponses should only contain 'name', 'id', and 'response' fields (NO 'error' field)
8. Follow the scenario and failure distributions exactly

Use the `save_eval_set` tool to save your output. Use the `validate_eval_set` tool to check your JSON before saving.
"""


def _format_scenario_distribution(gen_config: TestGenConfig) -> str:
    total = gen_config.total_test_cases_per_task
    lines = []
    for sw in gen_config.scenario_weights:
        count = round(total * sw.weight / 100)
        lines.append(f"- {sw.name}: {sw.weight}% (~{count} test cases)")
    return "\n".join(lines) if lines else "- (use default distribution)"


def _format_failure_distribution(gen_config: TestGenConfig) -> str:
    lines = []
    for fw in gen_config.failure_weights:
        lines.append(f"- {fw.name}: {fw.weight}%")
    return "\n".join(lines) if lines else "- (use default distribution)"


def _format_multi_turn_config(gen_config: TestGenConfig) -> str:
    mt = gen_config.multi_turn
    if not mt.enabled:
        return "Multi-turn test cases: DISABLED"
    parts = [
        f"- Enabled: yes",
        f"- Turn range: {mt.min_turns}-{mt.max_turns} turns",
    ]
    turn_types = []
    if mt.include_clarification:
        turn_types.append("clarification (user asks for more details)")
    if mt.include_correction:
        turn_types.append("correction (user corrects earlier input)")
    if mt.include_follow_up:
        turn_types.append("follow-up (user asks related question)")
    if turn_types:
        parts.append(f"- Turn types: {', '.join(turn_types)}")
    return "\n".join(parts)


def build_testcase_system_instruction(
    metadata: AgentMetadata,
    config: TestCaseConfig,
    gen_config: Optional[TestGenConfig] = None,
) -> str:
    """Build system instruction for the test case generator."""
    gen_config = gen_config or TestGenConfig()
    agent_tree = _format_agent_tree(metadata)
    metrics = ", ".join(f"{k}={v}" for k, v in config.eval_metrics.items())
    return SYSTEM_INSTRUCTION_TEMPLATE.format(
        agent_tree=agent_tree,
        metrics=metrics,
        match_type=config.tool_trajectory_match_type,
        judge_model=config.judge_model,
        total_per_task=gen_config.total_test_cases_per_task,
        scenario_distribution=_format_scenario_distribution(gen_config),
        failure_distribution=_format_failure_distribution(gen_config),
        multi_turn_config=_format_multi_turn_config(gen_config),
    )
