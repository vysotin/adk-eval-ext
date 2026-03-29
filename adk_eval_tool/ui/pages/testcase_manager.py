"""Page: Generate, view, and edit evaluation test cases with trajectory support."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import TestCaseConfig, TestGenConfig, MultiTurnConfig, ScenarioWeight
from adk_eval_tool.ui.components.json_editor import json_editor
from adk_eval_tool.ui.output_dir import get_output_path


def render():
    st.header("Test Cases (EvalSets)")

    if st.session_state.metadata is None:
        st.warning("No agent loaded. Launch with: `python -m adk_eval_tool <module> <variable>`")
        return

    tab_gen, tab_edit, tab_versions = st.tabs(["Generate", "View / Edit", "Dataset Versions"])

    with tab_gen:
        _render_generate_tab()

    with tab_edit:
        _render_edit_tab()

    with tab_versions:
        _render_versions_tab()


def _render_generate_tab():
    st.subheader("Generate Test Cases")

    if st.session_state.task_set is None:
        st.warning("Generate tasks first (Tasks & Scenarios page).")
        return

    task_set = st.session_state.task_set

    # --- Total & task selection ---
    st.markdown("#### Total Test Cases")

    task_options = {t.task_id: t.name for t in task_set.tasks}
    selected = st.multiselect(
        "Generate for tasks (all if empty)",
        options=list(task_options.keys()),
        format_func=lambda x: f"{x} — {task_options[x]}",
        key="tc_task_selection",
    )
    tasks_to_process = (
        [t for t in task_set.tasks if t.task_id in selected]
        if selected
        else task_set.tasks
    )
    num_tasks = len(tasks_to_process)

    col_total, col_model, col_match = st.columns(3)
    with col_total:
        total_test_cases = st.number_input(
            "Total test cases (all tasks)",
            min_value=1, max_value=500, value=max(num_tasks * 10, 10),
            help="Total number of test cases distributed across all selected tasks",
            key="tc_total",
        )
    with col_model:
        judge_model = st.selectbox(
            "Generator model",
            ["gemini-2.5-flash", "gemini-2.5-pro"],
            key="tc_judge_model",
        )
    with col_match:
        match_type = st.selectbox(
            "Tool trajectory match type",
            ["IN_ORDER", "EXACT", "ANY_ORDER"],
            key="tc_match_type",
        )

    # Compute per-task distribution
    if num_tasks > 0:
        base_per_task = total_test_cases // num_tasks
        remainder = total_test_cases % num_tasks
        per_task_counts = [base_per_task + (1 if i < remainder else 0) for i in range(num_tasks)]
    else:
        per_task_counts = []

    st.divider()

    # --- Scenario types with weights ---
    st.markdown("#### Path Types & Weights")
    st.caption("Distribute test cases across path types. Weights should sum to 100%.")

    all_scenario_types = [
        "happy_path", "failure_path", "edge_case", "multi_turn",
        "boundary_value", "concurrent_tool_use",
    ]
    default_scenario_weights = {"happy_path": 30, "failure_path": 30, "edge_case": 20, "multi_turn": 20}

    scenario_weights: list[ScenarioWeight] = []
    scenario_cols = st.columns([2, 1, 1])
    scenario_cols[0].markdown("**Path type**")
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
        st.warning(f"Path type weights sum to {total_scenario_weight}%, should be 100%.")

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

    # --- Multi-turn test cases ---
    st.markdown("#### Multi-Turn Settings")
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

    # --- User Simulations ---
    st.markdown("#### User Simulations")
    num_simulations = st.number_input(
        "User simulations per task",
        min_value=0, max_value=20, value=3,
        help="Additional dynamic conversation simulations per task (uses ADK user simulator)",
        key="tc_num_sims",
    )

    st.divider()

    # --- Distribution preview ---
    st.markdown("#### Distribution Preview")
    if num_tasks > 0 and scenario_weights and total_scenario_weight == 100:
        preview_data = []
        for i, task in enumerate(tasks_to_process):
            tc_count = per_task_counts[i]
            row = {"Task": f"{task.name} ({task.task_id})", "Cases": tc_count}
            for sw in scenario_weights:
                row[sw.name] = max(1, round(tc_count * sw.weight / 100))
            preview_data.append(row)
        preview_data.append({
            "Task": "**Total**",
            "Cases": total_test_cases,
            **{sw.name: sum(r[sw.name] for r in preview_data) for sw in scenario_weights},
        })
        st.dataframe(preview_data, use_container_width=True, hide_index=True)
    elif num_tasks > 0:
        for i, task in enumerate(tasks_to_process):
            st.caption(f"- {task.name}: {per_task_counts[i]} test cases")
    else:
        st.caption("No tasks selected.")

    st.divider()

    save_dir = st.text_input("Save directory", value=get_output_path("eval_datasets"))

    if st.button("Generate Test Cases"):
        with st.spinner("Generating evaluation datasets..."):
            from adk_eval_tool.testcase_generator import generate_test_cases

            config = TestCaseConfig(
                eval_metrics={"tool_trajectory_avg_score": 0.8},
                judge_model=judge_model,
                tool_trajectory_match_type=match_type,
            )

            results = []
            progress = st.progress(0)
            for idx, task in enumerate(tasks_to_process):
                tc_count = per_task_counts[idx] if idx < len(per_task_counts) else 10
                gen_config = TestGenConfig(
                    total_test_cases_per_task=tc_count,
                    multi_turn=multi_turn,
                    scenario_weights=scenario_weights,
                    failure_weights=failure_weights,
                    num_simulations_per_task=num_simulations,
                    judge_model=judge_model,
                    tool_trajectory_match_type=match_type,
                )
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
            st.success(f"Generated {len(results)} eval sets with {total_test_cases} total test cases")

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
    save_dir = st.text_input("Save directory", value=get_output_path("eval_datasets"), key="save_dir_edit")
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


# ---------------------------------------------------------------------------
# Dataset Versions tab
# ---------------------------------------------------------------------------

def _versions_dir() -> str:
    return get_output_path("versions")


def _render_versions_tab():
    st.subheader("Create New Version")

    if not st.session_state.eval_sets and not st.session_state.task_set:
        st.warning("No data to version. Generate tasks or test cases first.")
        return

    version_name = st.text_input(
        "Version name",
        value=datetime.now().strftime("%Y%m%d_%H%M%S"),
        key="version_name",
    )
    version_notes = st.text_area("Version notes", placeholder="What changed...", key="version_notes")
    source_dir = st.text_input("Source eval_datasets directory", value=get_output_path("eval_datasets"), key="version_src_dir")

    if st.button("Create Version", key="create_version"):
        version_path = Path(_versions_dir()) / version_name
        version_path.mkdir(parents=True, exist_ok=True)

        if st.session_state.metadata:
            (version_path / "metadata.json").write_text(
                st.session_state.metadata.model_dump_json(indent=2)
            )
        if st.session_state.task_set:
            (version_path / "tasks.json").write_text(
                st.session_state.task_set.model_dump_json(indent=2)
            )

        src = Path(source_dir)
        if src.exists():
            evalsets_dir = version_path / "evalsets"
            evalsets_dir.mkdir(exist_ok=True)
            for f in src.glob("*.evalset.json"):
                shutil.copy2(f, evalsets_dir / f.name)

        if st.session_state.eval_sets:
            evalsets_dir = version_path / "evalsets"
            evalsets_dir.mkdir(exist_ok=True)
            for es in st.session_state.eval_sets:
                eid = es.get("evalSetId", "unknown")
                (evalsets_dir / f"{eid}.evalset.json").write_text(json.dumps(es, indent=2))

        manifest = {
            "version": version_name,
            "created_at": datetime.now().isoformat(),
            "notes": version_notes,
            "agent_name": st.session_state.metadata.name if st.session_state.metadata else None,
            "num_tasks": len(st.session_state.task_set.tasks) if st.session_state.task_set else 0,
            "num_eval_sets": len(st.session_state.eval_sets),
        }
        (version_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        st.success(f"Version '{version_name}' created at {version_path}")

    st.divider()

    # Browse existing versions
    st.subheader("Browse Versions")
    versions_path = Path(_versions_dir())
    if not versions_path.exists():
        st.info("No versions created yet.")
        return

    versions = sorted([d for d in versions_path.iterdir() if d.is_dir()], reverse=True)
    if not versions:
        st.info("No versions found.")
        return

    for version_dir in versions:
        manifest_file = version_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {"version": version_dir.name}

        with st.expander(
            f"v{manifest.get('version', version_dir.name)} — "
            f"{manifest.get('created_at', 'unknown')} — "
            f"{manifest.get('notes', '')[:50]}"
        ):
            st.json(manifest)

            evalsets_dir = version_dir / "evalsets"
            if evalsets_dir.exists():
                st.markdown("**Eval Sets:**")
                for f in evalsets_dir.glob("*.evalset.json"):
                    st.markdown(f"- `{f.name}`")

            col_load, col_delete = st.columns(2)
            with col_load:
                if st.button("Load version", key=f"load_{version_dir.name}"):
                    _load_version(version_dir)
                    st.success(f"Loaded version {version_dir.name}")
                    st.rerun()
            with col_delete:
                if st.button("Delete version", key=f"del_{version_dir.name}", type="secondary"):
                    shutil.rmtree(version_dir)
                    st.success(f"Deleted version {version_dir.name}")
                    st.rerun()


def _load_version(version_dir: Path):
    from adk_eval_tool.schemas import AgentMetadata, TaskScenarioSet

    meta_file = version_dir / "metadata.json"
    if meta_file.exists():
        st.session_state.metadata = AgentMetadata.model_validate(json.loads(meta_file.read_text()))

    tasks_file = version_dir / "tasks.json"
    if tasks_file.exists():
        st.session_state.task_set = TaskScenarioSet.model_validate(json.loads(tasks_file.read_text()))

    evalsets_dir = version_dir / "evalsets"
    if evalsets_dir.exists():
        st.session_state.eval_sets = [
            json.loads(f.read_text()) for f in sorted(evalsets_dir.glob("*.evalset.json"))
        ]
