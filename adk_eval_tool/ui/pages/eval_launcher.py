"""Page: Launch evaluation runs and monitor progress."""

from __future__ import annotations

import streamlit as st


def render():
    st.header("Run Evaluation")

    if st.session_state.eval_run_config is None:
        st.warning("Configure evaluation first (Eval Config page).")
        return
    if not st.session_state.eval_sets:
        st.warning("Generate test cases first (Test Cases page).")
        return

    config = st.session_state.eval_run_config

    st.subheader("Run Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agent", config.agent_module)
    with col2:
        st.metric("Metrics", len(config.metrics))
    with col3:
        st.metric("Runs per case", config.num_runs)

    eval_set_options = {
        es.get("evalSetId", f"set_{i}"): es.get("name", es.get("evalSetId", f"set_{i}"))
        for i, es in enumerate(st.session_state.eval_sets)
    }
    selected_ids = st.multiselect(
        "Run for eval sets (all if empty)",
        options=list(eval_set_options.keys()),
        format_func=lambda x: eval_set_options[x],
    )

    sets_to_run = (
        [es for es in st.session_state.eval_sets if es.get("evalSetId") in selected_ids]
        if selected_ids
        else st.session_state.eval_sets
    )

    st.info(
        f"Will run **{len(sets_to_run)}** eval set(s) x "
        f"**{config.num_runs}** run(s) each. "
        f"Agent calls real tools during evaluation."
    )

    if st.button("Launch Evaluation", type="primary", disabled=not config.agent_module):
        import asyncio

        from adk_eval_tool.eval_runner import run_evaluation, ResultStore

        result_store = ResultStore()
        progress = st.progress(0, text="Starting evaluation...")

        try:
            results = asyncio.run(run_evaluation(
                config=config,
                eval_sets=sets_to_run,
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

            st.subheader("Results Summary")
            for result in results:
                status_icon = {"PASSED": "\u2705", "FAILED": "\u274c", "NOT_EVALUATED": "\u26aa"}.get(result.status, "\u26aa")
                scores_str = ", ".join(
                    f"{k}={v:.2f}" if v is not None else f"{k}=N/A"
                    for k, v in result.overall_scores.items()
                )
                st.markdown(f"{status_icon} **{result.eval_id}** -- {scores_str}")

        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback
            st.code(traceback.format_exc())
