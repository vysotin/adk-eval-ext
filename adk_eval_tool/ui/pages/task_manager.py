"""Page: Generate, view, and edit tasks & trajectories."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import TaskTrajectorySet, Task, Trajectory, TrajectoryStep
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Tasks & Trajectories")

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
    st.subheader("Generate Tasks & Base Trajectories")

    constraints = st.text_area(
        "User constraints / context",
        placeholder="e.g., Focus on multi-tool tasks, include all sub-agents...",
        key="task_constraints",
    )
    model = st.selectbox("Generator model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])

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
        with st.spinner("Generating tasks and trajectories..."):
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
            st.session_state.task_set = TaskTrajectorySet.model_validate(data)
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

    st.subheader(f"Agent: {task_set.agent_name} -- {len(task_set.tasks)} tasks")

    # --- Per-task editing ---
    tasks_to_delete: list[int] = []

    for i, task in enumerate(task_set.tasks):
        with st.expander(
            f"Task: {task.name} ({task.task_id}) -- {len(task.trajectories)} trajectories",
            expanded=False,
        ):
            # Header row with delete button
            col_info, col_del = st.columns([5, 1])
            with col_del:
                if st.button("Delete Task", key=f"del_task_{i}", type="secondary"):
                    tasks_to_delete.append(i)

            # Editable fields
            with col_info:
                new_name = st.text_input("Name", value=task.name, key=f"tname_{i}")
                new_desc = st.text_area("Description", value=task.description, key=f"tdesc_{i}", height=80)
                if new_name != task.name or new_desc != task.description:
                    task.name = new_name
                    task.description = new_desc

            st.divider()

            # --- Per-trajectory editing within this task ---
            trajs_to_delete: list[int] = []

            for j, traj in enumerate(task.trajectories):
                st.markdown(f"**Trajectory: {traj.name}** (`{traj.trajectory_id}`)")

                col_traj, col_traj_del = st.columns([5, 1])
                with col_traj_del:
                    if st.button("Delete", key=f"del_traj_{i}_{j}", type="secondary"):
                        trajs_to_delete.append(j)

                with col_traj:
                    traj_name = st.text_input("Trajectory name", value=traj.name, key=f"trname_{i}_{j}")
                    traj_desc = st.text_input("Trajectory description", value=traj.description, key=f"trdesc_{i}_{j}")
                    if traj_name != traj.name:
                        traj.name = traj_name
                    if traj_desc != traj.description:
                        traj.description = traj_desc

                # Steps as structured editors
                for k, step in enumerate(traj.steps):
                    with st.container():
                        st.caption(f"Step {k + 1}")
                        cols = st.columns([3, 2])
                        with cols[0]:
                            new_msg = st.text_area(
                                "User message", value=step.user_message,
                                key=f"step_msg_{i}_{j}_{k}", height=60,
                            )
                            step.user_message = new_msg
                        with cols[1]:
                            tools_str = ", ".join(step.expected_tool_calls)
                            new_tools_str = st.text_input(
                                "Expected tool calls (comma-separated)",
                                value=tools_str,
                                key=f"step_tools_{i}_{j}_{k}",
                            )
                            step.expected_tool_calls = [
                                t.strip() for t in new_tools_str.split(",") if t.strip()
                            ]

                # Add step button
                if st.button("Add Step", key=f"add_step_{i}_{j}"):
                    traj.steps.append(TrajectoryStep(user_message=""))
                    st.rerun()

                st.markdown("---")

            # Delete marked trajectories
            if trajs_to_delete:
                for idx in sorted(trajs_to_delete, reverse=True):
                    task.trajectories.pop(idx)
                st.rerun()

            # Add trajectory button
            st.markdown("**Add trajectory to this task:**")
            col_traj_id, col_traj_add = st.columns([3, 1])
            with col_traj_id:
                new_traj_name = st.text_input(
                    "New trajectory name",
                    placeholder="e.g., Basic search flow",
                    key=f"new_traj_name_{i}",
                )
            with col_traj_add:
                st.markdown("")  # spacing
                if st.button("Add Trajectory", key=f"add_traj_{i}"):
                    traj_id = new_traj_name.lower().replace(" ", "_") if new_traj_name else f"traj_{uuid.uuid4().hex[:6]}"
                    task.trajectories.append(Trajectory(
                        trajectory_id=traj_id,
                        name=new_traj_name or traj_id,
                        steps=[TrajectoryStep(user_message="")],
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

    # Delete marked tasks
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
            trajectories=[],
        ))
        st.session_state.task_set = task_set
        st.rerun()

    # --- Save / Download ---
    st.divider()
    col_save, col_download = st.columns(2)
    with col_save:
        save_path = st.text_input("Save path", value=f"{task_set.agent_name}_tasks.json", key="task_save_path")
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
