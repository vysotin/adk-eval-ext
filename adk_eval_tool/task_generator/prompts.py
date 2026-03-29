"""Prompt templates for the task/scenario generator agent."""

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


SYSTEM_INSTRUCTION_TEMPLATE = """You are an expert test designer for AI agents. Your job is to analyze an agent's capabilities and generate a comprehensive set of tasks and their test scenarios.

## Agent Under Test

{agent_tree}

## Your Task

Given the agent metadata above and any user constraints, generate a set of tasks (user goals the agent can handle) and for each task, define test scenarios. Each scenario describes a combination of input type (intent variation) and expected output type in plain text.

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
      "scenarios": [
        {{{{
          "scenario_id": "string (snake_case)",
          "name": "string",
          "description": "Plain text describing the input type (user intent variation) and expected output type"
        }}}}
      ]
    }}}}
  ],
  "generation_context": "string"
}}}}
```

## Guidelines

1. **Task Coverage**: Identify ALL distinct user tasks the agent can handle — every meaningful action a user would ask for.

2. **Scenario Design**: For each task, define scenarios that vary by:
   - Input type: different ways a user might express the intent
   - Output type: what kind of response is expected (e.g. list, summary, confirmation)
   - Do NOT specify exact conversation turns — those are generated at the test case stage.

3. **Tool Coverage**: Ensure every tool and sub-agent is referenced in at least one scenario description.

4. Use the `save_output` tool to save your final output.
"""


def build_system_instruction(metadata: AgentMetadata) -> str:
    """Build the system instruction with agent metadata embedded."""
    agent_tree = _format_agent_tree(metadata)
    return SYSTEM_INSTRUCTION_TEMPLATE.format(agent_tree=agent_tree)
