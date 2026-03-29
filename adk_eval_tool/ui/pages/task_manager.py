"""Page: Generate, view, and edit tasks & scenarios."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import TaskScenarioSet, Task, Scenario
from adk_eval_tool.ui.components.json_editor import json_editor
from adk_eval_tool.ui.output_dir import get_output_path


def render():
    st.header("Tasks & Scenarios")

    if st.session_state.metadata is None:
        st.warning("No agent loaded. Launch with: `python -m adk_eval_tool <module> <variable>`")
        return

    tab_gen, tab_edit = st.tabs(["Generate", "View / Edit"])

    with tab_gen:
        _render_generate_tab()

    with tab_edit:
        _render_edit_tab()


# ---------------------------------------------------------------------------
# Generate tab
# ---------------------------------------------------------------------------

def _render_generate_tab():
    st.subheader("Generate Tasks & Scenarios")

    constraints = st.text_area(
        "User constraints / context",
        placeholder="e.g., Focus on multi-tool tasks, include all sub-agents...",
        key="task_constraints",
    )
    model = st.selectbox("Generator model", ["gemini-2.5-flash", "gemini-2.5-pro"])

    if st.session_state.task_set and st.session_state.task_set.tasks:
        st.divider()
        st.markdown("**Selective regeneration:**")
        existing_tasks = [t.task_id for t in st.session_state.task_set.tasks]
        selected_tasks = st.multiselect(
            "Regenerate only these tasks (leave empty for full generation)",
            options=existing_tasks,
        )
    else:
        selected_tasks = []

    if st.button("Generate", key="gen_tasks"):
        with st.spinner("Generating tasks and scenarios..."):
            from adk_eval_tool.task_generator import generate_tasks

            try:
                result = asyncio.run(generate_tasks(
                    metadata=st.session_state.metadata,
                    user_constraints=constraints,
                    model=model,
                ))

                if selected_tasks and st.session_state.task_set:
                    existing = st.session_state.task_set
                    new_map = {t.task_id: t for t in result.tasks}
                    merged = []
                    for task in existing.tasks:
                        if task.task_id in selected_tasks and task.task_id in new_map:
                            merged.append(new_map[task.task_id])
                        else:
                            merged.append(task)
                    existing_ids = {t.task_id for t in existing.tasks}
                    for task in result.tasks:
                        if task.task_id not in existing_ids:
                            merged.append(task)
                    existing.tasks = merged
                    st.session_state.task_set = existing
                else:
                    st.session_state.task_set = result

                st.success(f"Generated {len(st.session_state.task_set.tasks)} tasks")
            except Exception as e:
                st.error(f"Generation failed: {e}")

    st.divider()
    uploaded = st.file_uploader("Or upload existing tasks JSON", type=["json"], key="task_upload")
    if uploaded:
        try:
            data = json.loads(uploaded.read())
            st.session_state.task_set = TaskScenarioSet.model_validate(data)
            st.success(f"Loaded {len(st.session_state.task_set.tasks)} tasks")
        except Exception as e:
            st.error(f"Failed to load: {e}")


# ---------------------------------------------------------------------------
# Edit tab
# ---------------------------------------------------------------------------

def _render_edit_tab():
    if st.session_state.task_set is None:
        st.warning("Generate or load tasks first.")
        return

    task_set = st.session_state.task_set

    st.subheader(f"Agent: {task_set.agent_name} — {len(task_set.tasks)} tasks")

    tasks_to_delete: list[int] = []

    for i, task in enumerate(task_set.tasks):
        with st.expander(
            f"Task: {task.name} ({task.task_id}) — {len(task.scenarios)} scenarios",
            expanded=False,
        ):
            col_info, col_del = st.columns([5, 1])
            with col_del:
                if st.button("Delete Task", key=f"del_task_{i}", type="secondary"):
                    tasks_to_delete.append(i)

            with col_info:
                new_name = st.text_input("Name", value=task.name, key=f"tname_{i}")
                new_desc = st.text_area("Description", value=task.description, key=f"tdesc_{i}", height=80)
                if new_name != task.name or new_desc != task.description:
                    task.name = new_name
                    task.description = new_desc

            st.divider()

            # --- Per-scenario editing ---
            scenarios_to_delete: list[int] = []

            for j, scenario in enumerate(task.scenarios):
                st.markdown(f"**Scenario: {scenario.name}** (`{scenario.scenario_id}`)")

                col_sc, col_sc_del = st.columns([5, 1])
                with col_sc_del:
                    if st.button("Delete", key=f"del_sc_{i}_{j}", type="secondary"):
                        scenarios_to_delete.append(j)

                with col_sc:
                    sc_name = st.text_input("Scenario name", value=scenario.name, key=f"scname_{i}_{j}")
                    sc_desc = st.text_area(
                        "Description (input type + expected output type)",
                        value=scenario.description,
                        key=f"scdesc_{i}_{j}",
                        height=100,
                    )
                    if sc_name != scenario.name:
                        scenario.name = sc_name
                    if sc_desc != scenario.description:
                        scenario.description = sc_desc

                st.markdown("---")

            if scenarios_to_delete:
                for idx in sorted(scenarios_to_delete, reverse=True):
                    task.scenarios.pop(idx)
                st.rerun()

            # Add scenario button
            st.markdown("**Add scenario to this task:**")
            col_sc_id, col_sc_add = st.columns([3, 1])
            with col_sc_id:
                new_sc_name = st.text_input(
                    "New scenario name",
                    placeholder="e.g., Flight search with valid dates",
                    key=f"new_sc_name_{i}",
                )
            with col_sc_add:
                st.markdown("")
                if st.button("Add Scenario", key=f"add_sc_{i}"):
                    sc_id = new_sc_name.lower().replace(" ", "_") if new_sc_name else f"sc_{uuid.uuid4().hex[:6]}"
                    task.scenarios.append(Scenario(
                        scenario_id=sc_id,
                        name=new_sc_name or sc_id,
                        description="",
                    ))
                    st.rerun()

            # Raw JSON fallback
            with st.expander("Raw JSON"):
                edited = json_editor(task.model_dump(), key=f"task_json_{i}", height=300)
                if edited != task.model_dump():
                    try:
                        task_set.tasks[i] = Task.model_validate(edited)
                        st.session_state.task_set = task_set
                    except Exception as e:
                        st.error(f"Invalid task data: {e}")

    if tasks_to_delete:
        for idx in sorted(tasks_to_delete, reverse=True):
            task_set.tasks.pop(idx)
        st.session_state.task_set = task_set
        st.rerun()

    # --- Add new task ---
    st.divider()
    st.subheader("Add New Task")
    col_new_id, col_new_name = st.columns(2)
    with col_new_id:
        new_task_id = st.text_input("Task ID (snake_case)", placeholder="e.g., book_flight", key="new_task_id")
    with col_new_name:
        new_task_name = st.text_input("Task name", placeholder="e.g., Book Flight", key="new_task_name")
    new_task_desc = st.text_input("Task description", placeholder="e.g., User wants to book a flight", key="new_task_desc")

    if st.button("Add Task", key="add_task_btn", type="primary"):
        task_id = new_task_id or f"task_{uuid.uuid4().hex[:6]}"
        task_name = new_task_name or task_id
        task_set.tasks.append(Task(
            task_id=task_id,
            name=task_name,
            description=new_task_desc,
            scenarios=[],
        ))
        st.session_state.task_set = task_set
        st.rerun()

    # --- Save / Download ---
    st.divider()
    col_save, col_download = st.columns(2)
    with col_save:
        default_path = get_output_path("tasks", f"{task_set.agent_name}_tasks.json")
        save_path = st.text_input("Save path", value=default_path, key="task_save_path")
        if st.button("Save to disk", key="save_tasks"):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text(task_set.model_dump_json(indent=2))
            st.success(f"Saved to {save_path}")
    with col_download:
        st.download_button(
            "Download JSON",
            data=task_set.model_dump_json(indent=2),
            file_name=f"{task_set.agent_name}_tasks.json",
            mime="application/json",
        )
