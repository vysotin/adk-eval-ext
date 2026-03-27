"""Main Streamlit application."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ADK Eval Tool",
    layout="wide",
)


def main():
    st.sidebar.title("ADK Eval Tool")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Agent Metadata",
            "Intents & Scenarios",
            "Test Cases",
            "Eval Config",
            "Run Evaluation",
            "Eval Results",
            "Dataset Versions",
        ],
    )

    # Session state initialization
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "intent_set" not in st.session_state:
        st.session_state.intent_set = None
    if "eval_sets" not in st.session_state:
        st.session_state.eval_sets = []
    if "eval_run_config" not in st.session_state:
        st.session_state.eval_run_config = None
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []

    if page == "Agent Metadata":
        from adk_eval_tool.ui.pages.metadata_viewer import render
        render()
    elif page == "Intents & Scenarios":
        from adk_eval_tool.ui.pages.intent_manager import render
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
