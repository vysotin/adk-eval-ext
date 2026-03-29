"""Page: Configure eval metrics, run scoring, view results.

Only metrics that work with GOOGLE_API_KEY (no GCP project required).

Tabs:
  1. Configure & Score — metric config, launch evaluation scoring
  2. Evaluation Results — scores, per-invocation breakdown, traces
"""

from __future__ import annotations

import statistics

import streamlit as st

from adk_eval_tool.schemas import (
    EvalRunConfig,
    EvalRunResult,
    InferenceRunResult,
    MetricConfig,
)
from adk_eval_tool.ui.output_dir import get_output_path


# Metrics that work with GOOGLE_API_KEY (no GCP project/Vertex AI needed).
# Excluded: safety_v1 (requires Vertex AI), per_turn_user_simulator_quality_v1 (requires Vertex AI)
BUILTIN_METRICS = {
    "tool_trajectory_avg_score": {
        "description": "Tool call sequence matches expected trajectory",
        "has_match_type": True,
        "default_threshold": 0.8,
    },
    "response_match_score": {
        "description": "ROUGE-1 unigram overlap with reference answer",
        "has_match_type": False,
        "default_threshold": 0.8,
    },
    "response_evaluation_score": {
        "description": "General response quality score (LLM judge)",
        "has_match_type": False,
        "default_threshold": 0.7,
    },
    "final_response_match_v2": {
        "description": "LLM-judged semantic equivalence to reference",
        "has_match_type": False,
        "default_threshold": 0.8,
    },
    "rubric_based_final_response_quality_v1": {
        "description": "LLM-judged quality against custom rubric",
        "has_match_type": False,
        "has_rubric": True,
        "default_threshold": 0.8,
    },
    "rubric_based_tool_use_quality_v1": {
        "description": "LLM-judged tool usage quality against rubric",
        "has_match_type": False,
        "has_rubric": True,
        "default_threshold": 0.8,
    },
    "hallucinations_v1": {
        "description": "Response grounded in tool outputs (no hallucinations)",
        "has_match_type": False,
        "has_evaluate_intermediate": True,
        "default_threshold": 0.9,
    },
}


def render():
    st.header("Evaluation")

    tab_config, tab_results = st.tabs(["Configure & Score", "Evaluation Results"])

    with tab_config:
        _render_config_and_score()
    with tab_results:
        _render_eval_results()


# ===================================================================
# Tab 1: Configure & Score
# ===================================================================

def _render_config_and_score():
    _render_metric_config()

    if st.session_state.eval_run_config is None:
        return

    st.divider()
    _render_scoring_section()


def _render_metric_config():
    st.subheader("Evaluation Metrics Configuration")
    st.caption("Only metrics compatible with GOOGLE_API_KEY are shown (no GCP project required).")

    col1, col2 = st.columns(2)
    with col1:
        judge_model = st.selectbox(
            "Judge model (for LLM-based metrics)",
            ["gemini-2.5-flash", "gemini-2.5-pro"],
            key="eval_judge_model",
        )
    with col2:
        eval_results_dir = st.text_input(
            "Results output directory",
            value=get_output_path("eval_results"),
            key="eval_results_save_dir",
        )

    st.divider()
    st.subheader("Metrics")

    metrics: list[MetricConfig] = []
    for metric_name, info in BUILTIN_METRICS.items():
        col_enable, col_config = st.columns([1, 3])
        with col_enable:
            enabled = st.checkbox(
                metric_name,
                value=metric_name == "tool_trajectory_avg_score",
                key=f"enable_{metric_name}",
            )
        if not enabled:
            continue

        with col_config:
            st.caption(info["description"])
            subcols = st.columns(3)
            with subcols[0]:
                threshold = st.slider(
                    "Threshold", 0.0, 1.0,
                    value=info["default_threshold"],
                    key=f"thresh_{metric_name}",
                )
            match_type = None
            if info.get("has_match_type"):
                with subcols[1]:
                    match_type = st.selectbox(
                        "Match type", ["IN_ORDER", "EXACT", "ANY_ORDER"],
                        key=f"match_{metric_name}",
                    )
            evaluate_intermediate = False
            if info.get("has_evaluate_intermediate"):
                with subcols[1]:
                    evaluate_intermediate = st.checkbox(
                        "Evaluate intermediate responses",
                        key=f"intermediate_{metric_name}",
                    )
            rubric = None
            if info.get("has_rubric"):
                rubric = st.text_area(
                    "Rubric", placeholder="Define quality criteria...",
                    key=f"rubric_{metric_name}",
                )
            metrics.append(MetricConfig(
                metric_name=metric_name,
                threshold=threshold,
                match_type=match_type,
                judge_model=judge_model,
                evaluate_intermediate=evaluate_intermediate,
                rubric=rubric if rubric else None,
            ))

    st.divider()

    # Build config from inference config + metrics
    agent_module = ""
    agent_name = None
    num_runs = 2
    trace_db = get_output_path("traces", "eval_traces.db")
    try:
        agent_module = st.session_state["_eval_agent_module"]
    except (KeyError, AttributeError):
        try:
            agent_module = st.session_state["preloaded_agent_module"]
        except (KeyError, AttributeError):
            pass
    try:
        agent_name = st.session_state["_eval_agent_name"] or None
    except (KeyError, AttributeError):
        pass

    if st.button("Save Configuration", type="primary"):
        eval_config = EvalRunConfig(
            agent_module=agent_module,
            agent_name=agent_name,
            metrics=metrics,
            judge_model=judge_model,
            num_runs=num_runs,
            trace_db_path=trace_db,
        )
        st.session_state.eval_run_config = eval_config
        st.success(f"Saved config: {len(metrics)} metrics, judge={judge_model}")

    if st.session_state.eval_run_config:
        st.subheader("Current Config")
        st.json(st.session_state.eval_run_config.model_dump())


def _render_scoring_section():
    st.subheader("Run Evaluation Scoring")

    inf_results: list[InferenceRunResult] = []
    try:
        inf_results = st.session_state["inference_results"]
    except (KeyError, AttributeError):
        pass

    if not inf_results:
        st.warning("Run inference first (Inference page).")
        return

    config = st.session_state.eval_run_config

    col1, col2 = st.columns([3, 1])
    with col1:
        eval_results_dir = st.session_state.get("eval_results_save_dir", get_output_path("eval_results"))
    with col2:
        st.metric("Inference results", len(inf_results))

    st.info(f"Will score **{len(inf_results)}** inference result(s) against **{len(config.metrics)}** metric(s).")

    if st.button("Launch Evaluation", type="primary", key="btn_launch_eval"):
        import asyncio
        from adk_eval_tool.eval_runner import run_eval_scoring, ResultStore

        result_store = ResultStore(base_dir=eval_results_dir)
        progress = st.progress(0, text="Scoring...")
        try:
            results = asyncio.run(run_eval_scoring(
                config=config,
                inference_results=inf_results,
                save_dir=eval_results_dir,
                result_store=result_store,
            ))
            st.session_state.eval_results = results
            progress.progress(1.0, text="Evaluation complete!")

            passed = sum(1 for r in results if r.status == "PASSED")
            failed = sum(1 for r in results if r.status == "FAILED")
            if failed == 0:
                st.success(f"All {passed} eval cases PASSED")
            else:
                st.error(f"{failed} FAILED, {passed} PASSED out of {len(results)} cases")
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ===================================================================
# Tab 2: Evaluation Results
# ===================================================================

def _render_eval_results():
    results: list[EvalRunResult] = []
    try:
        results = st.session_state["eval_results"]
    except (KeyError, AttributeError):
        pass

    if not results:
        st.info("No evaluation results yet. Run an evaluation first.")
        return

    st.subheader("Overall Summary")
    passed = sum(1 for r in results if r.status == "PASSED")
    failed = sum(1 for r in results if r.status == "FAILED")
    total = len(results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", total)
    col2.metric("Passed", passed, delta=f"{passed/total*100:.0f}%" if total else "0%")
    col3.metric("Failed", failed, delta=f"-{failed}" if failed else "0", delta_color="inverse")

    st.subheader("Average Scores")
    all_metrics: dict[str, list[float]] = {}
    for result in results:
        for metric, score in result.overall_scores.items():
            if score is not None:
                all_metrics.setdefault(metric, []).append(score)

    if all_metrics:
        metric_cols = st.columns(min(len(all_metrics), 4))
        for i, (metric, scores) in enumerate(all_metrics.items()):
            avg = statistics.mean(scores)
            metric_cols[i % len(metric_cols)].metric(metric, f"{avg:.3f}", delta=f"n={len(scores)}")

    st.divider()

    st.subheader("Per-Case Results")
    for result in results:
        status_icon = {"PASSED": "\u2705", "FAILED": "\u274c", "NOT_EVALUATED": "\u26aa"}.get(result.status, "\u26aa")
        with st.expander(
            f"{status_icon} {result.eval_id} — {result.status}",
            expanded=(result.status == "FAILED"),
        ):
            st.markdown("**Overall Scores:**")
            score_cols = st.columns(min(len(result.overall_scores) or 1, 4))
            for i, (metric, score) in enumerate(result.overall_scores.items()):
                score_str = f"{score:.3f}" if score is not None else "N/A"
                score_cols[i % len(score_cols)].metric(metric, score_str)

            if result.per_invocation_scores:
                st.markdown("**Per-Invocation Breakdown:**")
                for inv in result.per_invocation_scores:
                    inv_id = inv.get("invocation_id", "?")
                    user_msg = inv.get("user_message", "")
                    actual_resp = inv.get("actual_response", "")
                    expected_resp = inv.get("expected_response", "")
                    inv_scores = inv.get("scores", {})

                    with st.expander(f"Turn: {inv_id} — {user_msg[:60]}"):
                        st.markdown(f"**User:** {user_msg}")
                        actual_tools = inv.get("actual_tool_calls", [])
                        expected_tools = inv.get("expected_tool_calls", [])
                        if actual_tools or expected_tools:
                            col_exp_t, col_act_t = st.columns(2)
                            with col_exp_t:
                                st.markdown("**Expected Tool Calls:**")
                                for t in expected_tools:
                                    st.code(str(t), language="text")
                                if not expected_tools:
                                    st.caption("(none)")
                            with col_act_t:
                                st.markdown("**Actual Tool Calls:**")
                                for t in actual_tools:
                                    st.code(str(t), language="text")
                                if not actual_tools:
                                    st.caption("(none)")

                        col_exp, col_act = st.columns(2)
                        with col_exp:
                            st.markdown("**Expected Response:**")
                            st.text(expected_resp or "(none)")
                        with col_act:
                            st.markdown("**Actual Response:**")
                            st.text(actual_resp or "(none)")

                        if inv_scores:
                            st.markdown("**Scores:**")
                            for metric, score in inv_scores.items():
                                score_str = f"{score:.3f}" if score is not None else "N/A"
                                st.markdown(f"- {metric}: {score_str}")

            if result.basic_metrics:
                bm = result.basic_metrics
                st.markdown("**Basic Metrics:**")
                bm_cols = st.columns(5)
                bm_cols[0].metric("Total Tokens", f"{bm.total_tokens:,}")
                bm_cols[1].metric("Input Tokens", f"{bm.total_input_tokens:,}")
                bm_cols[2].metric("Output Tokens", f"{bm.total_output_tokens:,}")
                bm_cols[3].metric("LLM Calls", bm.num_llm_calls)
                bm_cols[4].metric("Tool Calls", bm.num_tool_calls)

            if result.trace_tree:
                st.markdown("**Execution Trace:**")
                try:
                    from adk_eval_tool.ui.components.trace_tree import render_trace_summary, render_trace_tree
                    render_trace_summary([result.trace_tree])
                    render_trace_tree(result.trace_tree)
                except ImportError:
                    st.caption("Trace tree component not available.")
