"""Prompt templates for the task/trajectory generator agent."""

from __future__ import annotations

from adk_eval_tool.schemas import AgentMetadata


def _format_tools(metadata: AgentMetadata, indent: int = 0) -> str:
    """Format tool list for prompt."""
    prefix = "  " * indent
    lines = []
    for tool in metadata.tools:
        lines.append(f"{prefix}- {tool.name}: {tool.description}")
        if tool.parameters_schema.get("properties"):
            for param, info in tool.parameters_schema["properties"].items():
                lines.append(f"{prefix}  param: {param} ({info.get('type', 'any')})")
    return "\n".join(lines)


def _format_agent_tree(metadata: AgentMetadata, indent: int = 0) -> str:
    """Format agent tree recursively."""
    prefix = "  " * indent
    lines = [
        f"{prefix}Agent: {metadata.name} ({metadata.agent_type})",
        f"{prefix}  Description: {metadata.description}",
        f"{prefix}  Instruction: {metadata.instruction[:200]}{'...' if len(metadata.instruction) > 200 else ''}",
        f"{prefix}  Model: {metadata.model}",
        f"{prefix}  Tools:",
    ]
    if metadata.tools:
        lines.append(_format_tools(metadata, indent + 2))
    else:
        lines.append(f"{prefix}    (none)")

    if metadata.sub_agents:
        lines.append(f"{prefix}  Sub-agents:")
        for sub in metadata.sub_agents:
            lines.append(_format_agent_tree(sub, indent + 2))

    return "\n".join(lines)


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert test designer for AI agents. Your job is to analyze an agent's capabilities and generate a comprehensive set of tasks and their base (happy-path) trajectories.

## Agent Under Test

{agent_tree}

## Your Task

Given the agent metadata above and any user constraints, generate a set of tasks (user goals the agent can handle) and for each task, define the base trajectory — the expected happy-path tool call sequence.

Only generate base/happy-path trajectories. Failure paths and edge cases will be generated at the test case generation stage.

## Output Requirements

You MUST output valid JSON matching this exact structure:

```json
{{{{
  "agent_name": "string",
  "tasks": [
    {{{{
      "task_id": "string (snake_case)",
      "name": "string",
      "description": "string",
      "trajectories": [
        {{{{
          "trajectory_id": "string (snake_case)",
          "name": "string",
          "description": "string",
          "steps": [
            {{{{
              "user_message": "string (the actual user message)",
              "expected_tool_calls": ["tool_name_1", "tool_name_2"],
              "expected_tool_args": {{{{"tool_name": {{{{"arg": "value"}}}}}}}},
              "expected_response_keywords": ["keyword1", "keyword2"],
              "notes": "string"
            }}}}
          ]
        }}}}
      ]
    }}}}
  ],
  "generation_context": "string"
}}}}
```

## Guidelines

1. **Task Coverage**: Identify ALL distinct user tasks the agent can handle — every meaningful action a user would ask for.

2. **Trajectory Design**: For each task, define the base happy-path trajectory:
   - The expected sequence of tool calls
   - Realistic user messages
   - Expected tool arguments
   - Only the successful/normal flow

3. **Tool Coverage**: Ensure every tool and sub-agent is exercised by at least one trajectory.

4. Use the `save_output` tool to save your final output.
"""


def build_system_instruction(metadata: AgentMetadata) -> str:
    """Build the system instruction with agent metadata embedded."""
    agent_tree = _format_agent_tree(metadata)
    return SYSTEM_INSTRUCTION_TEMPLATE.format(agent_tree=agent_tree)
