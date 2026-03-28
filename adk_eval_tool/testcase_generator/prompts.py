"""Prompt templates for the test case generator agent."""

from __future__ import annotations

from adk_eval_tool.task_generator.prompts import _format_agent_tree
from adk_eval_tool.schemas import AgentMetadata, TestCaseConfig


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert at creating evaluation datasets for Google ADK agents. You generate golden test cases in the ADK `.evalset.json` format.

Given a task and its base trajectory (happy path), you generate both happy-path and failure-path test cases.

## Agent Under Test

{agent_tree}

## Evaluation Config

- Metrics: {metrics}
- Tool trajectory match type: {match_type}
- Judge model: {judge_model}

## ADK EvalSet JSON Format

You MUST produce JSON in this exact structure (camelCase keys):

```json
{{{{
  "evalSetId": "<agent_name>__<task_id>",
  "name": "<Human-readable name>",
  "description": "<What this eval set tests>",
  "evalCases": [
    {{{{
      "evalId": "<task_id>__<trajectory_id>",
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
4. `evalId` format: `<task_id>__<trajectory_id>`
5. `evalSetId` format: `<agent_name>__<task_id>`
6. Generate both happy-path and failure-path test cases from the base trajectory
7. Multi-turn trajectories have multiple entries in the `conversation` array

Use the `save_eval_set` tool to save your output. Use the `validate_eval_set` tool to check your JSON before saving.
"""


def build_testcase_system_instruction(
    metadata: AgentMetadata,
    config: TestCaseConfig,
) -> str:
    """Build system instruction for the test case generator."""
    agent_tree = _format_agent_tree(metadata)
    metrics = ", ".join(f"{k}={v}" for k, v in config.eval_metrics.items())
    return SYSTEM_INSTRUCTION_TEMPLATE.format(
        agent_tree=agent_tree,
        metrics=metrics,
        match_type=config.tool_trajectory_match_type,
        judge_model=config.judge_model,
    )
