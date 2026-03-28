"""Page: Generate, view, and edit evaluation test cases with trajectory support."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import TestCaseConfig, TestGenConfig, MultiTurnConfig, ScenarioWeight
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Test Cases (EvalSets)")

    if st.session_state.metadata is None:
        st.warning("No agent loaded. Launch with: `python -m adk_eval_tool <module> <variable>`")
        return

    tab_gen, tab_edit = st.tabs(["Generate", "View / Edit"])

    with tab_gen:
        _render_generate_tab()

    with tab_edit:
        _render_edit_tab()


def _render_generate_tab():
    st.subheader("Generate Test Cases")

    if st.session_state.task_set is None:
        st.warning("Generate tasks first (Tasks & Trajectories page).")
        return

    # --- General settings ---
    st.markdown("#### General Settings")
    col_gen1, col_gen2, col_gen3 = st.columns(3)
    with col_gen1:
        total_per_task = st.number_input(
            "Total test cases per task",
            min_value=1, max_value=50, value=10,
            help="Total number of test cases to generate per task across all scenario types",
            key="tc_total_per_task",
        )
    with col_gen2:
        judge_model = st.selectbox(
            "Generator model",
            ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
            key="tc_judge_model",
        )
    with col_gen3:
        match_type = st.selectbox(
            "Tool trajectory match type",
            ["IN_ORDER", "EXACT", "ANY_ORDER"],
            key="tc_match_type",
        )

    st.divider()

    # --- Multi-turn test cases ---
    st.markdown("#### Multi-Turn Test Cases")
    enable_multi_turn = st.checkbox("Enable multi-turn test cases", value=True, key="tc_multi_turn_enabled")

    multi_turn = MultiTurnConfig(enabled=enable_multi_turn)
    if enable_multi_turn:
        col_mt1, col_mt2 = st.columns(2)
        with col_mt1:
            mt_min = st.number_input("Min turns", min_value=2, max_value=10, value=2, key="tc_mt_min")
            mt_max = st.number_input("Max turns", min_value=2, max_value=20, value=5, key="tc_mt_max")
        with col_mt2:
            mt_clarification = st.checkbox("Include clarification turns", value=True, key="tc_mt_clarification")
            mt_correction = st.checkbox("Include correction turns", value=True, key="tc_mt_correction")
            mt_follow_up = st.checkbox("Include follow-up turns", value=True, key="tc_mt_followup")
        multi_turn = MultiTurnConfig(
            enabled=True,
            min_turns=mt_min,
            max_turns=mt_max,
            include_clarification=mt_clarification,
            include_correction=mt_correction,
            include_follow_up=mt_follow_up,
        )

    st.divider()

    # --- Scenario types with weights ---
    st.markdown("#### Scenario Types & Weights")
    st.caption(f"Distribute {total_per_task} test cases across scenario types. Weights should sum to 100%.")

    all_scenario_types = [
        "happy_path", "failure_path", "edge_case", "multi_turn",
        "boundary_value", "concurrent_tool_use",
    ]
    default_scenario_weights = {"happy_path": 30, "failure_path": 30, "edge_case": 20, "multi_turn": 20}

    scenario_weights: list[ScenarioWeight] = []
    scenario_cols = st.columns([2, 1, 1])
    scenario_cols[0].markdown("**Scenario type**")
    scenario_cols[1].markdown("**Enabled**")
    scenario_cols[2].markdown("**Weight %**")

    for stype in all_scenario_types:
        cols = st.columns([2, 1, 1])
        with cols[1]:
            enabled = st.checkbox(
                stype, value=stype in default_scenario_weights,
                key=f"tc_scen_enable_{stype}", label_visibility="collapsed",
            )
        with cols[0]:
            st.markdown(f"`{stype}`")
        if enabled:
            with cols[2]:
                weight = st.number_input(
                    f"{stype} weight", min_value=0, max_value=100,
                    value=default_scenario_weights.get(stype, 10),
                    key=f"tc_scen_weight_{stype}", label_visibility="collapsed",
                )
            scenario_weights.append(ScenarioWeight(name=stype, weight=weight))

    total_scenario_weight = sum(sw.weight for sw in scenario_weights)
    if scenario_weights and total_scenario_weight != 100:
        st.warning(f"Scenario weights sum to {total_scenario_weight}%, should be 100%.")
    elif scenario_weights:
        # Show distribution preview
        preview_parts = []
        for sw in scenario_weights:
            count = round(total_per_task * sw.weight / 100)
            preview_parts.append(f"{sw.name}: ~{count}")
        st.caption(f"Distribution: {', '.join(preview_parts)}")

    st.divider()

    # --- Failure types with weights ---
    st.markdown("#### Failure Types & Weights")
    st.caption("Distribute failure_path test cases across failure types. Weights should sum to 100%.")

    all_failure_types = [
        "missing_required_input", "invalid_input_format", "tool_error",
        "ambiguous_request", "unauthorized_access", "timeout",
        "out_of_scope_request", "partial_input",
    ]
    default_failure_weights = {
        "missing_required_input": 25, "invalid_input_format": 25,
        "tool_error": 25, "ambiguous_request": 25,
    }

    failure_weights: list[ScenarioWeight] = []
    fail_header_cols = st.columns([2, 1, 1])
    fail_header_cols[0].markdown("**Failure type**")
    fail_header_cols[1].markdown("**Enabled**")
    fail_header_cols[2].markdown("**Weight %**")

    for ftype in all_failure_types:
        cols = st.columns([2, 1, 1])
        with cols[1]:
            enabled = st.checkbox(
                ftype, value=ftype in default_failure_weights,
                key=f"tc_fail_enable_{ftype}", label_visibility="collapsed",
            )
        with cols[0]:
            st.markdown(f"`{ftype}`")
        if enabled:
            with cols[2]:
                weight = st.number_input(
                    f"{ftype} weight", min_value=0, max_value=100,
                    value=default_failure_weights.get(ftype, 10),
                    key=f"tc_fail_weight_{ftype}", label_visibility="collapsed",
                )
            failure_weights.append(ScenarioWeight(name=ftype, weight=weight))

    total_failure_weight = sum(fw.weight for fw in failure_weights)
    if failure_weights and total_failure_weight != 100:
        st.warning(f"Failure weights sum to {total_failure_weight}%, should be 100%.")

    st.divider()

    # --- User Simulations ---
    st.markdown("#### User Simulations")
    num_simulations = st.number_input(
        "User simulations per task",
        min_value=0, max_value=20, value=3,
        help="Additional dynamic conversation simulations per task (uses ADK user simulator)",
        key="tc_num_sims",
    )

    st.divider()

    # --- Build configs ---
    gen_config = TestGenConfig(
        total_test_cases_per_task=total_per_task,
        multi_turn=multi_turn,
        scenario_weights=scenario_weights,
        failure_weights=failure_weights,
        num_simulations_per_task=num_simulations,
        judge_model=judge_model,
        tool_trajectory_match_type=match_type,
    )

    config = TestCaseConfig(
        eval_metrics={
            "tool_trajectory_avg_score": 0.8,
            "safety_v1": 1.0,
        },
        judge_model=judge_model,
        tool_trajectory_match_type=match_type,
    )

    # --- Task selection ---
    task_set = st.session_state.task_set
    task_options = {t.task_id: t.name for t in task_set.tasks}
    selected = st.multiselect(
        "Generate for tasks (all if empty)",
        options=list(task_options.keys()),
        format_func=lambda x: f"{x} -- {task_options[x]}",
    )

    save_dir = st.text_input("Save directory", value="eval_datasets")

    if st.button("Generate Test Cases"):
        with st.spinner("Generating evaluation datasets..."):
            from adk_eval_tool.testcase_generator import generate_test_cases

            tasks_to_process = (
                [t for t in task_set.tasks if t.task_id in selected]
                if selected
                else task_set.tasks
            )

            results = []
            progress = st.progress(0)
            for idx, task in enumerate(tasks_to_process):
                try:
                    eval_set = asyncio.run(generate_test_cases(
                        metadata=st.session_state.metadata,
                        task=task,
                        config=config,
                        gen_config=gen_config,
                        save_dir=save_dir,
                    ))
                    results.append(eval_set)
                except Exception as e:
                    st.error(f"Failed for task {task.task_id}: {e}")
                progress.progress((idx + 1) / len(tasks_to_process))

            st.session_state.eval_sets = results
            st.success(f"Generated {len(results)} eval sets")

    st.divider()
    uploaded = st.file_uploader("Or upload existing .evalset.json", type=["json"], key="evalset_upload")
    if uploaded:
        try:
            data = json.loads(uploaded.read())
            if "evalSetId" in data:
                st.session_state.eval_sets.append(data)
                st.success(f"Loaded eval set: {data.get('evalSetId')}")
            else:
                st.error("File does not appear to be an ADK EvalSet (missing evalSetId)")
        except Exception as e:
            st.error(f"Failed to load: {e}")


def _render_edit_tab():
    if not st.session_state.eval_sets:
        st.warning("Generate or upload test cases first.")
        return

    for es_idx, eval_set in enumerate(st.session_state.eval_sets):
        eval_set_id = eval_set.get("evalSetId", f"evalset_{es_idx}")
        cases = eval_set.get("evalCases", [])

        with st.expander(f"EvalSet: {eval_set_id} -- {len(cases)} case(s)", expanded=True):
            col_name, col_desc = st.columns(2)
            with col_name:
                eval_set["name"] = st.text_input(
                    "Name", value=eval_set.get("name", ""), key=f"es_name_{es_idx}"
                )
            with col_desc:
                eval_set["description"] = st.text_input(
                    "Description", value=eval_set.get("description", ""), key=f"es_desc_{es_idx}"
                )

            cases_to_delete = []
            for case_idx, case in enumerate(cases):
                eval_id = case.get("evalId", f"case_{case_idx}")
                is_dynamic = "conversation_scenario" in case

                with st.container():
                    col_title, col_del = st.columns([5, 1])
                    with col_title:
                        icon = "\U0001f504" if is_dynamic else "\U0001f4cb"
                        st.markdown(f"#### {icon} Case: `{eval_id}`")
                    with col_del:
                        if st.button("Delete", key=f"del_case_{es_idx}_{case_idx}", type="secondary"):
                            cases_to_delete.append(case_idx)

                    if is_dynamic:
                        _render_conversation_scenario_editor(case, es_idx, case_idx)
                    else:
                        _render_trajectory_editor(case, es_idx, case_idx)

                    st.divider()

            for idx in sorted(cases_to_delete, reverse=True):
                cases.pop(idx)
                st.rerun()

            st.markdown("**Add new eval case:**")
            col_type, col_add = st.columns([2, 1])
            with col_type:
                new_case_type = st.radio(
                    "Type", ["Trajectory (static)", "Conversation Scenario (dynamic)"],
                    key=f"new_type_{es_idx}", horizontal=True,
                )
            with col_add:
                if st.button("Add Case", key=f"add_case_{es_idx}"):
                    new_id = f"new_case_{uuid.uuid4().hex[:6]}"
                    if "Trajectory" in new_case_type:
                        cases.append({
                            "evalId": new_id,
                            "conversation": [{
                                "invocationId": "inv-1",
                                "userContent": {"role": "user", "parts": [{"text": ""}]},
                                "intermediateData": {"toolUses": [], "toolResponses": [], "intermediateResponses": []},
                            }],
                        })
                    else:
                        cases.append({
                            "evalId": new_id,
                            "conversation_scenario": {"starting_prompt": "", "conversation_plan": ""},
                        })
                    st.rerun()

            with st.expander("Raw JSON editor"):
                edited = json_editor(eval_set, key=f"evalset_raw_{es_idx}")
                if edited != eval_set:
                    st.session_state.eval_sets[es_idx] = edited
                    st.success("EvalSet updated from raw JSON.")

    st.divider()
    save_dir = st.text_input("Save directory", value="eval_datasets", key="save_dir_edit")
    if st.button("Save all to disk", key="save_evalsets"):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        for eval_set in st.session_state.eval_sets:
            eid = eval_set.get("evalSetId", "unknown")
            (path / f"{eid}.evalset.json").write_text(json.dumps(eval_set, indent=2))
        st.success(f"Saved {len(st.session_state.eval_sets)} eval sets to {save_dir}/")


def _render_trajectory_editor(case: dict, es_idx: int, case_idx: int):
    """Render structured editor for trajectory-based eval cases."""
    conversation = case.get("conversation", [])

    session_input = case.get("sessionInput")
    if session_input:
        with st.expander("Session Input (initial state)"):
            state_str = json.dumps(session_input.get("state", {}), indent=2)
            new_state_str = st.text_area("State", value=state_str, key=f"sess_state_{es_idx}_{case_idx}")
            try:
                case["sessionInput"]["state"] = json.loads(new_state_str)
            except json.JSONDecodeError:
                pass

    for inv_idx, inv in enumerate(conversation):
        inv_id = inv.get("invocationId", f"inv-{inv_idx + 1}")
        st.markdown(f"**Turn {inv_idx + 1}** (`{inv_id}`)")

        user_text = ""
        if inv.get("userContent", {}).get("parts"):
            user_text = inv["userContent"]["parts"][0].get("text", "")
        new_user_text = st.text_area(
            "User message", value=user_text, key=f"user_{es_idx}_{case_idx}_{inv_idx}",
        )
        inv["userContent"] = {"role": "user", "parts": [{"text": new_user_text}]}

        intermediate = inv.get("intermediateData", {})

        col_tools, col_responses = st.columns(2)
        with col_tools:
            st.markdown("**Expected tool calls** (toolUses)")
            tool_uses = intermediate.get("toolUses", [])
            tool_uses_str = json.dumps(tool_uses, indent=2)
            new_tool_uses_str = st.text_area(
                "toolUses", value=tool_uses_str, height=120,
                key=f"tool_uses_{es_idx}_{case_idx}_{inv_idx}",
                help='[{"name": "tool_name", "args": {"key": "value"}}]',
            )
            try:
                intermediate["toolUses"] = json.loads(new_tool_uses_str)
            except json.JSONDecodeError:
                pass

        with col_responses:
            st.markdown("**Reference tool responses** (for hallucination eval)")
            tool_responses = intermediate.get("toolResponses", [])
            tool_resp_str = json.dumps(tool_responses, indent=2)
            new_tool_resp_str = st.text_area(
                "toolResponses", value=tool_resp_str, height=120,
                key=f"tool_resp_{es_idx}_{case_idx}_{inv_idx}",
                help='[{"name": "tool", "id": "call_1", "response": {...}}]',
            )
            try:
                intermediate["toolResponses"] = json.loads(new_tool_resp_str)
            except json.JSONDecodeError:
                pass

        inv["intermediateData"] = intermediate

        inter_resps = intermediate.get("intermediateResponses", [])
        if inter_resps:
            with st.expander("Intermediate responses"):
                inter_str = json.dumps(inter_resps, indent=2)
                new_inter_str = st.text_area(
                    "intermediateResponses", value=inter_str,
                    key=f"inter_resp_{es_idx}_{case_idx}_{inv_idx}",
                )
                try:
                    intermediate["intermediateResponses"] = json.loads(new_inter_str)
                except json.JSONDecodeError:
                    pass

        final_text = ""
        if inv.get("finalResponse", {}).get("parts"):
            final_text = inv["finalResponse"]["parts"][0].get("text", "")
        new_final_text = st.text_area(
            "Expected response", value=final_text,
            key=f"final_{es_idx}_{case_idx}_{inv_idx}",
        )
        if new_final_text:
            inv["finalResponse"] = {"role": "model", "parts": [{"text": new_final_text}]}

        rubrics = inv.get("rubrics", [])
        rubric_text = ""
        if rubrics:
            rubric_text = rubrics[0].get("rubricContent", {}).get("textProperty", "")
        new_rubric = st.text_input(
            "Quality rubric (optional)", value=rubric_text,
            key=f"rubric_{es_idx}_{case_idx}_{inv_idx}",
        )
        if new_rubric:
            inv["rubrics"] = [{
                "rubricId": f"rubric_inv_{inv_idx + 1}",
                "rubricContent": {"textProperty": new_rubric},
            }]
        elif rubrics and not new_rubric:
            inv.pop("rubrics", None)

    if st.button("Add turn", key=f"add_turn_{es_idx}_{case_idx}"):
        conversation.append({
            "invocationId": f"inv-{len(conversation) + 1}",
            "userContent": {"role": "user", "parts": [{"text": ""}]},
            "intermediateData": {"toolUses": [], "toolResponses": [], "intermediateResponses": []},
        })
        st.rerun()


def _render_conversation_scenario_editor(case: dict, es_idx: int, case_idx: int):
    """Render editor for dynamic conversation_scenario eval cases."""
    cs = case.get("conversation_scenario", {})

    cs["starting_prompt"] = st.text_area(
        "Starting prompt", value=cs.get("starting_prompt", ""),
        key=f"cs_prompt_{es_idx}_{case_idx}",
    )
    cs["conversation_plan"] = st.text_area(
        "Conversation plan", value=cs.get("conversation_plan", ""),
        key=f"cs_plan_{es_idx}_{case_idx}",
        help="Instructions for the user simulator. End with 'Signal completion when done.'",
    )

    case["conversation_scenario"] = cs
