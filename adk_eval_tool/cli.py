"""CLI launcher for ADK Eval Tool.

Usage:
    python -m adk_eval_tool.cli <agent_module_path> <agent_variable_name>

Examples:
    python -m adk_eval_tool.cli examples.weather_agent.agent root_agent
    python -m adk_eval_tool.cli my_project.agent root_agent
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


def _resolve_module_to_path(module_path: str) -> str | None:
    """Resolve a dotted module path to a filesystem .py path."""
    parts = module_path.split(".")
    # Try as a direct path to a .py file within the current directory
    candidate = Path.cwd() / Path(*parts).with_suffix(".py")
    if candidate.exists():
        return str(candidate)
    # Try importlib to find the file
    try:
        spec = importlib.util.find_spec(module_path)
        if spec and spec.origin:
            return spec.origin
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def parse_agent_from_module(module_path: str, agent_var: str):
    """Parse the agent from live object (default) or source code (fallback).

    Returns:
        Tuple of (AgentMetadata, error_message). On success error_message is None.
        On failure AgentMetadata is None.
    """
    # Try live-object parser first (import + introspect)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        # Module not importable — try source-code parser
        source_path = _resolve_module_to_path(module_path)
        if source_path:
            from adk_eval_tool.agent_parser import parse_agent_from_source
            try:
                metadata = parse_agent_from_source(source_path, agent_variable=agent_var)
                return metadata, None
            except (ValueError, FileNotFoundError) as e2:
                return None, str(e2)
        return None, f"Module not found: {module_path}\n  {e}"
    except Exception as e:
        return None, f"Failed to import module '{module_path}': {e}"

    if not hasattr(module, agent_var):
        available = [a for a in dir(module) if not a.startswith("_")]
        return None, (
            f"Variable '{agent_var}' not found in module '{module_path}'.\n"
            f"  Available names: {', '.join(available[:20])}"
        )

    agent_obj = getattr(module, agent_var)

    from google.adk.agents.base_agent import BaseAgent
    if not isinstance(agent_obj, BaseAgent):
        return None, (
            f"'{agent_var}' in '{module_path}' is not an ADK BaseAgent instance.\n"
            f"  Got: {type(agent_obj).__name__}"
        )

    from adk_eval_tool.agent_parser import parse_agent
    try:
        metadata = parse_agent(agent_obj)
    except Exception as e:
        return None, f"Failed to parse agent: {e}"

    return metadata, None


def main():
    parser = argparse.ArgumentParser(
        prog="adk-eval-tool",
        description="Launch ADK Eval Tool UI with a pre-parsed agent.",
    )
    parser.add_argument(
        "agent_module",
        help="Python module path to the agent (e.g., examples.weather_agent.agent)",
    )
    parser.add_argument(
        "agent_variable",
        nargs="?",
        default="root_agent",
        help="Variable name referencing the root agent (default: root_agent)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501)",
    )

    args = parser.parse_args()

    # Load .env from current directory or project root
    _load_env()

    print(f"Parsing agent: {args.agent_module}:{args.agent_variable} ...", flush=True)

    metadata, error = parse_agent_from_module(args.agent_module, args.agent_variable)

    if error:
        print(f"\nError: {error}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"  Agent: {metadata.name} ({metadata.agent_type})")
    print(f"  Tools: {[t.name for t in metadata.tools]}")
    if metadata.sub_agents:
        print(f"  Sub-agents: {[a.name for a in metadata.sub_agents]}")
    print()

    # Write metadata to a output file that app.py will pick up
    metadata_json = metadata.model_dump_json(indent=2)
    tmpdir = tempfile.mkdtemp(prefix="adk_eval_")
    metadata_path = Path(tmpdir) / "preloaded_metadata.json"
    metadata_path.write_text(metadata_json)

    # Set env vars for Streamlit to pick up
    os.environ["ADK_EVAL_PRELOADED_METADATA"] = str(metadata_path)
    os.environ["ADK_EVAL_AGENT_MODULE"] = args.agent_module
    os.environ["ADK_EVAL_AGENT_VARIABLE"] = args.agent_variable

    print(f"Launching Streamlit UI on port {args.port} ...")
    print(f"  Agent metadata pre-loaded for: {metadata.name}")
    print()

    # Launch Streamlit
    app_path = Path(__file__).parent / "ui" / "app.py"
    os.execvp(
        sys.executable,
        [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", str(args.port),
            "--server.headless", "true",
            "--",  # separator for streamlit args
        ],
    )


def _load_env():
    """Load .env file from CWD or common locations."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path)
            return


if __name__ == "__main__":
    main()
