"""Main Streamlit application."""

from __future__ import annotations

import json
import os
from pathlib import Path

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

st.html("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
</style>
""")

PAGES = [
    "Agent Metadata",
    "Tasks & Trajectories",
    "Test Cases",
    "Inference",
    "Evaluation",
]


DEFAULT_OUTPUT_DIR = "./output"


def _init_session_state():
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = DEFAULT_OUTPUT_DIR
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "task_set" not in st.session_state:
        st.session_state.task_set = None
    if "eval_sets" not in st.session_state:
        st.session_state.eval_sets = []
    if "eval_run_config" not in st.session_state:
        st.session_state.eval_run_config = None
    if "inference_results" not in st.session_state:
        st.session_state.inference_results = []
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    if st.session_state.metadata is None:
        preloaded_path = os.environ.get("ADK_EVAL_PRELOADED_METADATA")
        if preloaded_path and Path(preloaded_path).exists():
            try:
                from adk_eval_tool.schemas import AgentMetadata

                data = json.loads(Path(preloaded_path).read_text())
                st.session_state.metadata = AgentMetadata.model_validate(data)

                agent_module = os.environ.get("ADK_EVAL_AGENT_MODULE", "")
                agent_variable = os.environ.get("ADK_EVAL_AGENT_VARIABLE", "root_agent")
                if "preloaded_agent_module" not in st.session_state:
                    st.session_state.preloaded_agent_module = agent_module
                    st.session_state.preloaded_agent_variable = agent_variable
            except Exception:
                pass


def _compute_max_step() -> int:
    """Compute the highest reachable step based on session state data."""
    max_step = 1  # 0-1 always available

    task_set = st.session_state.task_set
    if task_set and task_set.tasks:
        max_step = 2
    else:
        return max_step

    if st.session_state.eval_sets:
        max_step = 3
    else:
        return max_step

    if st.session_state.inference_results:
        max_step = 4
    else:
        return max_step

    return max_step


def _go_to_step(idx: int):
    st.session_state.current_step = idx


def _render_flow_nav():
    current = st.session_state.current_step
    max_reached = _compute_max_step()

    if current > max_reached:
        st.session_state.current_step = max_reached
        current = max_reached

    for i, page_name in enumerate(PAGES):
        is_current = i == current
        is_reachable = i <= max_reached
        is_completed = i < current and is_reachable

        if is_current:
            icon = "▶"
        elif is_completed:
            icon = "✓"
        else:
            icon = f"{i + 1}"

        label = f"{icon}  {page_name}"
        if is_reachable:
            st.sidebar.button(
                label, key=f"nav_step_{i}",
                on_click=_go_to_step, args=(i,),
                use_container_width=True,
            )
        else:
            st.sidebar.button(
                label, key=f"nav_step_{i}",
                disabled=True, use_container_width=True,
            )

        if i < len(PAGES) - 1:
            st.sidebar.markdown(
                "<div style='text-align:center;line-height:1;margin:-8px 0 -8px 0;font-size:1.2em;opacity:0.4;'>↓</div>",
                unsafe_allow_html=True,
            )


def main():
    _init_session_state()

    st.sidebar.title("Eval Engineering Tool")

    if st.session_state.metadata:
        meta = st.session_state.metadata
        st.sidebar.success(f"Agent: **{meta.name}**")
        st.sidebar.caption(
            f"{meta.agent_type} | "
            f"{len(meta.tools)} tools | "
            f"{len(meta.sub_agents)} sub-agents"
        )

    st.sidebar.markdown("---")
    st.session_state.output_dir = st.sidebar.text_input(
        "Output directory",
        value=st.session_state.output_dir,
        key="sidebar_output_dir",
    )

    _render_flow_nav()

    page = PAGES[st.session_state.current_step]

    if page == "Agent Metadata":
        from adk_eval_tool.ui.pages.metadata_viewer import render
        render()
    elif page == "Tasks & Trajectories":
        from adk_eval_tool.ui.pages.task_manager import render
        render()
    elif page == "Test Cases":
        from adk_eval_tool.ui.pages.testcase_manager import render
        render()
    elif page == "Inference":
        from adk_eval_tool.ui.pages.inference import render
        render()
    elif page == "Evaluation":
        from adk_eval_tool.ui.pages.evaluation import render
        render()


if __name__ == "__main__":
    main()
