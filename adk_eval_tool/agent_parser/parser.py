"""Parse ADK Agent metadata from Python source code using AST.

This parser extracts agent metadata by statically analysing the source file
rather than importing the module and inspecting live objects. It handles the
common ADK patterns:

- ``Agent(name=..., model=..., tools=[func_a, func_b], sub_agents=[...])``
- ``SequentialAgent / ParallelAgent / LoopAgent`` with sub_agents
- Tool functions defined in the same file
- Agent variables assigned at module level or in functions

The previous live-object parser is preserved as ``live_parser.py``.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any, Optional

from adk_eval_tool.schemas import AgentMetadata, ToolMetadata


# ADK agent class names we recognise in source code
_KNOWN_AGENT_CLASSES = {
    "Agent": "LlmAgent",
    "LlmAgent": "LlmAgent",
    "SequentialAgent": "SequentialAgent",
    "ParallelAgent": "ParallelAgent",
    "LoopAgent": "LoopAgent",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _literal_value(node: ast.AST) -> Any:
    """Try to evaluate a constant / simple literal from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_value(el) for el in node.elts]
    if isinstance(node, ast.Name):
        return node.id  # returns the variable name as a string reference
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.JoinedStr):
        return ast.unparse(node)
    return ast.unparse(node)


def _get_keyword(call: ast.Call, name: str) -> Any | None:
    """Return the value of a keyword argument *name* from an ``ast.Call``."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _get_keyword_value(call: ast.Call, name: str) -> Any | None:
    """Like ``_get_keyword`` but resolves to a literal value."""
    node = _get_keyword(call, name)
    if node is None:
        return None
    return _literal_value(node)


# ---------------------------------------------------------------------------
# Function / tool extraction
# ---------------------------------------------------------------------------


def _extract_function_schema_from_ast(func_node: ast.FunctionDef) -> dict[str, Any]:
    """Build a JSON-schema-like dict from a function's AST definition."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    for arg in func_node.args.args:
        param_name = arg.arg
        if param_name in ("self", "ctx", "tool_context", "context"):
            continue

        ann_str = ""
        if arg.annotation:
            ann_str = ast.unparse(arg.annotation)
            if "Context" in ann_str:
                continue

        prop: dict[str, Any] = {"type": type_map.get(ann_str, "string")}
        properties[param_name] = prop

    # Determine defaults — they apply to the *last N* args
    num_defaults = len(func_node.args.defaults)
    num_args = len(func_node.args.args)
    default_offset = num_args - num_defaults

    for i, arg in enumerate(func_node.args.args):
        if arg.arg in ("self", "ctx", "tool_context", "context"):
            continue
        ann_str = ast.unparse(arg.annotation) if arg.annotation else ""
        if "Context" in ann_str:
            continue

        if i >= default_offset:
            default_node = func_node.args.defaults[i - default_offset]
            try:
                properties[arg.arg]["default"] = _literal_value(default_node)
            except Exception:
                pass
        else:
            required.append(arg.arg)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _extract_docstring(node: ast.FunctionDef | ast.ClassDef) -> str:
    """Return the docstring of a function/class AST node, or empty string."""
    return ast.get_docstring(node) or ""


# ---------------------------------------------------------------------------
# Agent call parsing
# ---------------------------------------------------------------------------


def _parse_agent_call(
    call: ast.Call,
    functions: dict[str, ast.FunctionDef],
    agents: dict[str, AgentMetadata],
) -> AgentMetadata:
    """Parse a single ``Agent(...)`` / ``SequentialAgent(...)`` call node."""

    # Determine the agent class name
    if isinstance(call.func, ast.Name):
        class_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        class_name = call.func.attr
    else:
        class_name = "Agent"

    agent_type = _KNOWN_AGENT_CLASSES.get(class_name, class_name)

    name = _get_keyword_value(call, "name") or ""
    description = _get_keyword_value(call, "description") or ""
    model = _get_keyword_value(call, "model") or ""

    # Instruction — may be a string literal or multi-line
    instruction_node = _get_keyword(call, "instruction")
    instruction = ""
    if instruction_node is not None:
        if isinstance(instruction_node, ast.Constant) and isinstance(instruction_node.value, str):
            instruction = instruction_node.value
        else:
            instruction = ast.unparse(instruction_node)

    output_key = _get_keyword_value(call, "output_key")
    disallow_transfer_to_parent = _get_keyword_value(call, "disallow_transfer_to_parent") or False
    disallow_transfer_to_peers = _get_keyword_value(call, "disallow_transfer_to_peers") or False

    # Tools
    tools: list[ToolMetadata] = []
    tools_node = _get_keyword(call, "tools")
    if tools_node is not None and isinstance(tools_node, ast.List):
        for el in tools_node.elts:
            tool_name = _literal_value(el)
            if isinstance(tool_name, str) and tool_name in functions:
                func_node = functions[tool_name]
                tools.append(ToolMetadata(
                    name=tool_name,
                    description=_extract_docstring(func_node),
                    parameters_schema=_extract_function_schema_from_ast(func_node),
                    source="function",
                ))
            else:
                tools.append(ToolMetadata(
                    name=str(tool_name),
                    description="",
                    parameters_schema={},
                    source="function",
                ))

    # Sub-agents
    sub_agents: list[AgentMetadata] = []
    sub_agents_node = _get_keyword(call, "sub_agents")
    if sub_agents_node is not None and isinstance(sub_agents_node, ast.List):
        for el in sub_agents_node.elts:
            ref = _literal_value(el)
            if isinstance(ref, str) and ref in agents:
                sub_agents.append(agents[ref])
            elif isinstance(el, ast.Call):
                sub_agents.append(_parse_agent_call(el, functions, agents))

    return AgentMetadata(
        name=name,
        agent_type=agent_type,
        description=description,
        instruction=instruction,
        model=model,
        tools=tools,
        sub_agents=sub_agents,
        output_key=output_key,
        disallow_transfer_to_parent=disallow_transfer_to_parent if isinstance(disallow_transfer_to_parent, bool) else False,
        disallow_transfer_to_peers=disallow_transfer_to_peers if isinstance(disallow_transfer_to_peers, bool) else False,
    )


def _is_agent_call(node: ast.Call) -> bool:
    """Return True if the AST Call node looks like an ADK agent constructor."""
    if isinstance(node.func, ast.Name):
        return node.func.id in _KNOWN_AGENT_CLASSES
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in _KNOWN_AGENT_CLASSES
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_agent_from_source(
    source_path: str,
    agent_variable: str = "root_agent",
    save_path: Optional[str] = None,
) -> AgentMetadata:
    """Parse an ADK agent from Python source code into AgentMetadata.

    Statically analyses the source file using the ``ast`` module to extract
    agent definitions, tool functions, and sub-agent relationships without
    importing the module.

    Args:
        source_path: Path to the Python source file containing the agent.
        agent_variable: Name of the variable holding the root agent
            (default ``"root_agent"``).
        save_path: Optional file path to save the metadata JSON.

    Returns:
        AgentMetadata with the full recursive structure.

    Raises:
        FileNotFoundError: If *source_path* does not exist.
        ValueError: If *agent_variable* is not found or is not an agent call.
    """
    source_text = Path(source_path).read_text()
    tree = ast.parse(source_text, filename=source_path)

    # First pass: collect all top-level function definitions (potential tools)
    functions: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node

    # Second pass: collect all agent assignments in order (agents may
    # reference earlier agents as sub_agents)
    agents: dict[str, AgentMetadata] = {}
    target_metadata: AgentMetadata | None = None

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not node.targets or not isinstance(node.targets[0], ast.Name):
            continue
        var_name = node.targets[0].id
        value = node.value
        if not isinstance(value, ast.Call) or not _is_agent_call(value):
            continue

        metadata = _parse_agent_call(value, functions, agents)
        agents[var_name] = metadata

        if var_name == agent_variable:
            target_metadata = metadata

    if target_metadata is None:
        available = list(agents.keys())
        raise ValueError(
            f"Agent variable '{agent_variable}' not found in '{source_path}'.\n"
            f"  Available agent variables: {available}"
        )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(target_metadata.model_dump_json(indent=2))

    return target_metadata


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------------
# The previous default API (``parse_agent``) required a live object.
# We keep it available via ``live_parser`` for callers that still need it,
# and make ``parse_agent_from_source`` the recommended entry point.


def parse_agent(agent, save_path: Optional[str] = None) -> AgentMetadata:
    """Parse a live ADK agent object into an AgentMetadata tree.

    This delegates to the archived live-object parser.
    """
    from adk_eval_tool.agent_parser.live_parser import parse_agent as _live_parse
    return _live_parse(agent, save_path=save_path)


async def parse_agent_async(agent, save_path: Optional[str] = None) -> AgentMetadata:
    """Parse a live ADK agent (async, resolves MCP toolsets).

    This delegates to the archived live-object parser.
    """
    from adk_eval_tool.agent_parser.live_parser import parse_agent_async as _live_parse_async
    return await _live_parse_async(agent, save_path=save_path)
