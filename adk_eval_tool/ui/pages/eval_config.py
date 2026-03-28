"""Page: Configure evaluation metrics, thresholds, and judge model."""

from __future__ import annotations

import streamlit as st

from adk_eval_tool.schemas import EvalRunConfig, MetricConfig, UserSimulatorConfig, CustomMetricDef


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
    "safety_v1": {
        "description": "Response is safe and harmless",
        "has_match_type": False,
        "default_threshold": 1.0,
    },
    "per_turn_user_simulator_quality_v1": {
        "description": "User simulator fidelity (for conversation_scenario tests only)",
        "has_match_type": False,
        "default_threshold": 0.8,
    },
}


def render():
    st.header("Evaluation Configuration")

    st.subheader("Agent Module")
    col1, col2 = st.columns(2)
    with col1:
        default_module = st.session_state.get(
            "_eval_agent_module",
            st.session_state.get("preloaded_agent_module", ""),
        )
        agent_module = st.text_input(
            "Agent module path",
            value=default_module,
            placeholder="my_agent.agent",
            key="eval_agent_module",
        )
    with col2:
        agent_name = st.text_input(
            "Agent name (empty = root_agent)",
            value=st.session_state.get("_eval_agent_name", ""),
            key="eval_agent_name",
        )

    st.subheader("Judge Model & Runs")
    col1, col2, col3 = st.columns(3)
    with col1:
        judge_model = st.selectbox(
            "Judge model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            key="eval_judge_model",
        )
    with col2:
        num_runs = st.number_input("Number of runs", min_value=1, max_value=10, value=2, key="eval_num_runs")
    with col3:
        trace_db = st.text_input("Trace DB path", value="eval_traces.db", key="eval_trace_db")

    st.divider()
    st.subheader("Metrics")
    st.markdown("Enable metrics and configure thresholds. All 9 ADK built-in metrics are available.")

    metrics: list[MetricConfig] = []
    for metric_name, info in BUILTIN_METRICS.items():
        col_enable, col_config = st.columns([1, 3])
        with col_enable:
            enabled = st.checkbox(
                metric_name,
                value=metric_name in ("tool_trajectory_avg_score", "safety_v1"),
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
                        "Match type",
                        ["IN_ORDER", "EXACT", "ANY_ORDER"],
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
                    "Rubric",
                    placeholder="Define quality criteria...",
                    key=f"rubric_{metric_name}",
                )

            metrics.append(MetricConfig(
                metric_name=metric_name,
                threshold=threshold,
                match_type=match_type,
                evaluate_intermediate=evaluate_intermediate,
                rubric=rubric if rubric else None,
            ))

    # --- User Simulator Config ---
    st.divider()
    st.subheader("User Simulator (for conversation_scenario tests)")

    enable_user_sim = st.checkbox(
        "Enable user simulator",
        value=False,
        key="enable_user_sim",
        help="Required for conversation_scenario eval cases. Uses an LLM to simulate user turns.",
    )

    user_sim_config = None
    if enable_user_sim:
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            sim_model = st.selectbox(
                "Simulator model",
                ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                key="sim_model",
            )
            max_invocations = st.number_input(
                "Max invocations",
                min_value=1, max_value=50, value=20,
                key="sim_max_invocations",
                help="Maximum conversation turns before stopping (-1 for unlimited)",
            )
        with col_sim2:
            custom_instructions = st.text_area(
                "Custom instructions (optional)",
                placeholder="Must contain {{ stop_signal }}, {{ conversation_plan }}, {{ conversation_history }}",
                key="sim_custom_instructions",
                height=100,
            )
        user_sim_config = UserSimulatorConfig(
            model=sim_model,
            max_allowed_invocations=max_invocations,
            custom_instructions=custom_instructions if custom_instructions else None,
        )

    # --- Custom Metrics ---
    st.divider()
    st.subheader("Custom Metrics (optional)")

    custom_metrics: list[CustomMetricDef] = []
    num_custom = st.number_input(
        "Number of custom metrics",
        min_value=0, max_value=10, value=0,
        key="num_custom_metrics",
    )
    for i in range(int(num_custom)):
        with st.expander(f"Custom Metric {i + 1}", expanded=True):
            cm_name = st.text_input("Metric name", key=f"cm_name_{i}", placeholder="my_custom_metric")
            cm_path = st.text_input("Code path", key=f"cm_path_{i}", placeholder="my_module.my_metric_function")
            cm_desc = st.text_input("Description", key=f"cm_desc_{i}", placeholder="What this metric measures")
            if cm_name and cm_path:
                custom_metrics.append(CustomMetricDef(
                    name=cm_name, code_path=cm_path, description=cm_desc,
                ))

    # --- Save ---
    st.divider()

    if st.button("Save Configuration", type="primary"):
        eval_config = EvalRunConfig(
            agent_module=agent_module,
            agent_name=agent_name or None,
            metrics=metrics,
            judge_model=judge_model,
            num_runs=num_runs,
            trace_db_path=trace_db,
            user_simulator=user_sim_config,
            custom_metrics=custom_metrics,
        )
        st.session_state.eval_run_config = eval_config
        st.session_state._eval_agent_module = agent_module
        st.session_state._eval_agent_name = agent_name
        parts = [f"{len(metrics)} metrics"]
        if user_sim_config:
            parts.append("user simulator ON")
        if custom_metrics:
            parts.append(f"{len(custom_metrics)} custom metrics")
        st.success(f"Saved config: {', '.join(parts)}")

    if st.session_state.eval_run_config:
        config = st.session_state.eval_run_config
        st.subheader("Current Config")
        st.json(config.model_dump())
