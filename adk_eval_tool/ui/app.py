"""Main Streamlit application."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Load .env before anything else so GOOGLE_API_KEY is available
# for all ADK agent operations (intent generator, testcase generator, eval runner)
try:
    from dotenv import load_dotenv

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
except ImportError:
    pass

import streamlit as st

st.set_page_config(
    page_title="ADK Eval Tool",
    layout="wide",
)


def _init_session_state():
    """Initialize session state with defaults and any pre-loaded data."""
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "task_set" not in st.session_state:
        st.session_state.task_set = None
    if "eval_sets" not in st.session_state:
        st.session_state.eval_sets = []
    if "eval_run_config" not in st.session_state:
        st.session_state.eval_run_config = None
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []

    # Pick up pre-loaded metadata from CLI launcher (adk_eval_tool.cli)
    if st.session_state.metadata is None:
        preloaded_path = os.environ.get("ADK_EVAL_PRELOADED_METADATA")
        if preloaded_path and Path(preloaded_path).exists():
            try:
                from adk_eval_tool.schemas import AgentMetadata

                data = json.loads(Path(preloaded_path).read_text())
                st.session_state.metadata = AgentMetadata.model_validate(data)

                # Store the agent module info for eval config
                agent_module = os.environ.get("ADK_EVAL_AGENT_MODULE", "")
                agent_variable = os.environ.get("ADK_EVAL_AGENT_VARIABLE", "root_agent")
                if "preloaded_agent_module" not in st.session_state:
                    st.session_state.preloaded_agent_module = agent_module
                    st.session_state.preloaded_agent_variable = agent_variable
            except Exception:
                pass


def main():
    _init_session_state()

    st.sidebar.title("Eval Engineering Tool")

    # Show loaded agent info in sidebar
    if st.session_state.metadata:
        meta = st.session_state.metadata
        st.sidebar.success(f"Agent: **{meta.name}**")
        st.sidebar.caption(
            f"{meta.agent_type} | "
            f"{len(meta.tools)} tools | "
            f"{len(meta.sub_agents)} sub-agents"
        )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Agent Metadata",
            "Tasks & Trajectories",
            "Test Cases",
            "Eval Config",
            "Run Evaluation",
            "Eval Results",
            "Dataset Versions",
        ],
    )

    if page == "Agent Metadata":
        from adk_eval_tool.ui.pages.metadata_viewer import render
        render()
    elif page == "Tasks & Trajectories":
        from adk_eval_tool.ui.pages.task_manager import render
        render()
    elif page == "Test Cases":
        from adk_eval_tool.ui.pages.testcase_manager import render
        render()
    elif page == "Eval Config":
        from adk_eval_tool.ui.pages.eval_config import render
        render()
    elif page == "Run Evaluation":
        from adk_eval_tool.ui.pages.eval_launcher import render
        render()
    elif page == "Eval Results":
        from adk_eval_tool.ui.pages.eval_results import render
        render()
    elif page == "Dataset Versions":
        from adk_eval_tool.ui.pages.dataset_versions import render
        render()


if __name__ == "__main__":
    main()
