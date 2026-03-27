"""Page: Explore evaluation results, trace trees, and scores."""

from __future__ import annotations

import statistics

import streamlit as st

from adk_eval_tool.schemas import EvalRunResult


def render():
    st.header("Evaluation Results Explorer")

    tab_current, tab_history = st.tabs(["Current Run", "Result History"])

    with tab_current:
        _render_current_results()

    with tab_history:
        _render_result_history()


def _render_current_results():
    results: list[EvalRunResult] = st.session_state.get("eval_results", [])
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
            metric_cols[i % len(metric_cols)].metric(
                metric, f"{avg:.3f}", delta=f"n={len(scores)}",
            )

    st.divider()

    st.subheader("Per-Case Results")
    for result in results:
        status_icon = {"PASSED": "\u2705", "FAILED": "\u274c", "NOT_EVALUATED": "\u26aa"}.get(result.status, "\u26aa")
        with st.expander(
            f"{status_icon} {result.eval_id} -- {result.status}",
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

                    with st.expander(f"Turn: {inv_id} -- {user_msg[:60]}"):
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

            # Basic metrics
            if result.basic_metrics:
                bm = result.basic_metrics
                st.markdown("**Basic Metrics:**")
                bm_cols = st.columns(5)
                bm_cols[0].metric("Total Tokens", f"{bm.total_tokens:,}")
                bm_cols[1].metric("Input Tokens", f"{bm.total_input_tokens:,}")
                bm_cols[2].metric("Output Tokens", f"{bm.total_output_tokens:,}")
                bm_cols[3].metric("LLM Calls", bm.num_llm_calls)
                bm_cols[4].metric("Tool Calls", bm.num_tool_calls)

                bm_cols2 = st.columns(3)
                bm_cols2[0].metric("Duration", f"{bm.total_duration_ms:.0f}ms")
                bm_cols2[1].metric("Max Context", f"{bm.max_context_size:,} tokens")
                bm_cols2[2].metric("Avg Response", f"{bm.avg_response_length:.0f} chars")

            # Trace tree
            if result.trace_tree:
                st.markdown("**Execution Trace:**")
                try:
                    from adk_eval_tool.ui.components.trace_tree import render_trace_summary, render_trace_tree
                    render_trace_summary([result.trace_tree])
                    render_trace_tree(result.trace_tree)
                except ImportError:
                    st.caption("Trace tree component not available.")
            else:
                st.caption("No trace data available for this case.")


def _render_result_history():
    st.subheader("Historical Results")

    try:
        from adk_eval_tool.eval_runner import ResultStore
        result_store = ResultStore()
        all_results = result_store.load_results()
    except ImportError:
        st.info("Eval runner module not available yet.")
        return

    if not all_results:
        st.info("No historical results found.")
        return

    by_eval_set: dict[str, list[EvalRunResult]] = {}
    for r in all_results:
        by_eval_set.setdefault(r.eval_set_id, []).append(r)

    for eval_set_id, results in by_eval_set.items():
        with st.expander(f"EvalSet: {eval_set_id} -- {len(results)} runs"):
            averages = result_store.compute_averages(eval_set_id=eval_set_id)
            if averages:
                st.markdown("**Average Scores:**")
                avg_cols = st.columns(min(len(averages), 4))
                for i, (metric, avg) in enumerate(averages.items()):
                    avg_cols[i % len(avg_cols)].metric(metric, f"{avg:.3f}")

            passed = sum(1 for r in results if r.status == "PASSED")
            st.markdown(f"**Pass rate:** {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")

            for result in sorted(results, key=lambda r: r.timestamp, reverse=True):
                status_icon = {"PASSED": "\u2705", "FAILED": "\u274c"}.get(result.status, "\u26aa")
                scores_str = ", ".join(
                    f"{k}={v:.2f}" if v is not None else f"{k}=N/A"
                    for k, v in result.overall_scores.items()
                )
                st.markdown(f"  {status_icon} `{result.run_id}` -- {result.eval_id} -- {scores_str}")
