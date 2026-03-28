"""Prompt templates for the test case generator agent."""

from __future__ import annotations

from typing import Optional

from adk_eval_tool.task_generator.prompts import _format_agent_tree
from adk_eval_tool.schemas import AgentMetadata, TestCaseConfig, TestGenConfig


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert at creating evaluation datasets for Google ADK agents. You generate golden test cases in the ADK `.evalset.json` format.

Given a task and its base trajectory (happy path), you generate test cases covering multiple scenario types and failure modes.

## Agent Under Test

{agent_tree}

## Evaluation Config

- Metrics: {metrics}
- Tool trajectory match type: {match_type}
- Judge model: {judge_model}

## Test Generation Parameters

- Total simulations per task: {num_simulations}
- Scenario types to generate: {scenario_types}
- Failure types to simulate: {failure_types}

For each task, generate {num_simulations} total test cases covering the specified scenario types.
For failure paths, simulate these specific failure types: {failure_types}.

## ADK EvalSet JSON Format

You MUST produce JSON in this exact structure (camelCase keys):

```json
{{{{
  "evalSetId": "<agent_name>__<task_id>",
  "name": "<Human-readable name>",
  "description": "<What this eval set tests>",
  "evalCases": [
    {{{{
      "evalId": "<task_id>__<scenario_type>_<description>",
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

Use the `save_eval_set` tool to save your output. Use the `validate_eval_set` tool to check your JSON before saving.
"""


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
        num_simulations=gen_config.num_simulations_per_task,
        scenario_types=", ".join(gen_config.scenario_types),
        failure_types=", ".join(gen_config.failure_types),
    )
