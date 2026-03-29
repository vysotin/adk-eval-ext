"""Page: Run agent inference and view results.

Tabs:
  1. Run Inference — configure agent module, num_runs, output dir, launch
  2. Inference Results — read-only view of captured invocations
"""

from __future__ import annotations

import streamlit as st

from adk_eval_tool.schemas import EvalRunConfig, InferenceRunResult
from adk_eval_tool.ui.output_dir import get_output_path


def render():
    st.header("Inference")

    tab_run, tab_results = st.tabs(["Run Inference", "Inference Results"])

    with tab_run:
        _render_run_tab()
    with tab_results:
        _render_results_tab()


def _render_run_tab():
    if not st.session_state.eval_sets:
        st.warning("Generate test cases first (Test Cases page).")
        return

    st.subheader("Inference Configuration")

    col1, col2 = st.columns(2)
    with col1:
        default_module = ""
        try:
            default_module = st.session_state["_eval_agent_module"]
        except (KeyError, AttributeError):
            try:
                default_module = st.session_state["preloaded_agent_module"]
            except (KeyError, AttributeError):
                pass
        agent_module = st.text_input(
            "Agent module path",
            value=default_module,
            placeholder="my_agent.agent",
            key="eval_agent_module",
        )
    with col2:
        default_name = ""
        try:
            default_name = st.session_state["_eval_agent_name"]
        except (KeyError, AttributeError):
            pass
        agent_name = st.text_input(
            "Agent name (empty = root_agent)",
            value=default_name,
            key="eval_agent_name",
        )

    col_runs, col_trace, col_dir = st.columns(3)
    with col_runs:
        num_runs = st.number_input("Number of runs", min_value=1, max_value=10, value=2, key="inf_num_runs")
    with col_trace:
        trace_db = st.text_input("Trace DB path", value=get_output_path("traces", "eval_traces.db"), key="inf_trace_db")
    with col_dir:
        inference_dir = st.text_input("Output directory", value=get_output_path("inference"), key="inference_save_dir")

    st.divider()

    st.metric("Eval sets", len(st.session_state.eval_sets))
    st.info(
        f"Will run **{len(st.session_state.eval_sets)}** eval set(s) x "
        f"**{num_runs}** run(s) each."
    )

    if st.button("Launch Inference", type="primary", key="btn_launch_inference", disabled=not agent_module):
        import asyncio
        from adk_eval_tool.eval_runner import run_inference_only

        # Build a minimal config for inference (no metrics needed)
        config = EvalRunConfig(
            agent_module=agent_module,
            agent_name=agent_name or None,
            num_runs=num_runs,
            trace_db_path=trace_db,
        )
        st.session_state._eval_agent_module = agent_module
        st.session_state._eval_agent_name = agent_name

        progress = st.progress(0, text="Running inference...")
        try:
            results = asyncio.run(run_inference_only(
                config=config,
                eval_sets=st.session_state.eval_sets,
                save_dir=inference_dir,
            ))
            st.session_state.inference_results = results
            progress.progress(1.0, text="Inference complete!")
            st.success(f"Captured {len(results)} inference result(s)")
        except TimeoutError as e:
            progress.progress(1.0, text="Timed out")
            st.error(f"Inference timed out: {e}")
        except Exception as e:
            progress.progress(1.0, text="Failed")
            st.error(f"Inference error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_results_tab():
    inf_results: list[InferenceRunResult] = []
    try:
        inf_results = st.session_state["inference_results"]
    except (KeyError, AttributeError):
        pass

    if not inf_results:
        st.info("No inference results yet. Run inference first.")
        return

    st.subheader(f"Inference Results ({len(inf_results)} cases)")

    for result in inf_results:
        sid = result.session_id[:12] + "…" if result.session_id else "—"
        with st.expander(f"{result.eval_id} — session {sid}"):
            if result.actual_invocations:
                for inv in result.actual_invocations:
                    st.markdown(f"**User:** {inv.get('user_message', '')}")
                    st.markdown(f"**Response:** {inv.get('actual_response', '')}")
                    tools = inv.get("actual_tool_calls", [])
                    if tools:
                        st.markdown("**Tool calls:**")
                        for t in tools:
                            st.code(str(t), language="text")

            if result.basic_metrics:
                bm = result.basic_metrics
                cols = st.columns(4)
                cols[0].metric("Tokens", f"{bm.total_tokens:,}")
                cols[1].metric("LLM Calls", bm.num_llm_calls)
                cols[2].metric("Tool Calls", bm.num_tool_calls)
                cols[3].metric("Duration", f"{bm.total_duration_ms:.0f}ms")
