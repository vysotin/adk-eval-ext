"""Parse a live ADK Agent object into an AgentMetadata tree."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional, Union

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

from adk_eval_tool.schemas import AgentMetadata, ToolMetadata


_AGENT_TYPE_MAP = {
    LlmAgent: "LlmAgent",
    SequentialAgent: "SequentialAgent",
    ParallelAgent: "ParallelAgent",
    LoopAgent: "LoopAgent",
}


def _get_agent_type(agent: BaseAgent) -> str:
    for cls, name in _AGENT_TYPE_MAP.items():
        if isinstance(agent, cls):
            return name
    return type(agent).__name__


def _extract_function_schema(func: callable) -> dict[str, Any]:
    """Extract JSON schema from a Python function's signature."""
    sig = inspect.signature(func)
    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "ctx", "tool_context", "context"):
            continue
        ann = param.annotation
        if ann != inspect.Parameter.empty:
            ann_name = getattr(ann, "__name__", str(ann))
            if "Context" in ann_name:
                continue

        prop: dict[str, Any] = {}
        if ann != inspect.Parameter.empty:
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
            prop["type"] = type_map.get(ann, "string")
        else:
            prop["type"] = "string"

        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default

        properties[param_name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parse_tool(tool: Union[callable, BaseTool]) -> ToolMetadata:
    """Parse a single tool into ToolMetadata."""
    if isinstance(tool, BaseTool):
        return ToolMetadata(
            name=tool.name,
            description=tool.description or "",
            parameters_schema=_extract_declaration_schema(tool),
            source="function" if isinstance(tool, FunctionTool) else "builtin",
        )
    elif callable(tool):
        return ToolMetadata(
            name=tool.__name__,
            description=inspect.cleandoc(tool.__doc__ or ""),
            parameters_schema=_extract_function_schema(tool),
            source="function",
        )
    else:
        return ToolMetadata(name=str(tool), description="", parameters_schema={})


def _extract_declaration_schema(tool: BaseTool) -> dict[str, Any]:
    """Try to get schema from a BaseTool's declaration."""
    try:
        decl = tool._get_declaration()
        if decl and decl.parameters:
            schema = {"type": "object", "properties": {}}
            if hasattr(decl.parameters, "properties") and decl.parameters.properties:
                for prop_name, prop_schema in decl.parameters.properties.items():
                    schema["properties"][prop_name] = {
                        "type": getattr(prop_schema, "type", "string"),
                    }
            if hasattr(decl.parameters, "required") and decl.parameters.required:
                schema["required"] = list(decl.parameters.required)
            return schema
    except Exception:
        pass
    return {}


def _parse_tools_sync(tools_list: list) -> list[ToolMetadata]:
    """Parse tools synchronously. Toolsets that need async are deferred."""
    result = []
    for tool in tools_list:
        if isinstance(tool, BaseToolset):
            result.append(ToolMetadata(
                name=f"toolset:{type(tool).__name__}",
                description="Toolset requiring async resolution",
                parameters_schema={},
                source="mcp",
            ))
        else:
            result.append(_parse_tool(tool))
    return result


def _parse_agent_recursive(agent: BaseAgent) -> AgentMetadata:
    """Recursively parse an agent into AgentMetadata."""
    instruction = ""
    model = ""
    tools: list[ToolMetadata] = []
    output_key = None
    disallow_transfer_to_parent = False
    disallow_transfer_to_peers = False

    if isinstance(agent, LlmAgent):
        raw_instruction = agent.instruction
        if isinstance(raw_instruction, str):
            instruction = raw_instruction
        elif callable(raw_instruction):
            instruction = f"<dynamic: {raw_instruction.__name__}>"
        else:
            instruction = str(raw_instruction) if raw_instruction else ""

        model = agent.model if isinstance(agent.model, str) else str(agent.model) if agent.model else ""
        tools = _parse_tools_sync(agent.tools)
        output_key = agent.output_key
        disallow_transfer_to_parent = agent.disallow_transfer_to_parent
        disallow_transfer_to_peers = agent.disallow_transfer_to_peers

    sub_agents = [_parse_agent_recursive(sub) for sub in agent.sub_agents]

    return AgentMetadata(
        name=agent.name,
        agent_type=_get_agent_type(agent),
        description=agent.description or "",
        instruction=instruction,
        model=model,
        tools=tools,
        sub_agents=sub_agents,
        output_key=output_key,
        disallow_transfer_to_parent=disallow_transfer_to_parent,
        disallow_transfer_to_peers=disallow_transfer_to_peers,
    )


def parse_agent(
    agent: BaseAgent,
    save_path: Optional[str] = None,
) -> AgentMetadata:
    """Parse an ADK agent object into an AgentMetadata tree.

    Args:
        agent: A live ADK BaseAgent instance (Agent, SequentialAgent, etc.)
        save_path: Optional file path to save the metadata JSON.

    Returns:
        AgentMetadata with the full recursive structure.
    """
    metadata = _parse_agent_recursive(agent)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(metadata.model_dump_json(indent=2))

    return metadata
