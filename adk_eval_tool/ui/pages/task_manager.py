"""Page: Generate, view, and edit tasks & trajectories."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import TaskTrajectorySet, Task
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Tasks & Trajectories")

    if st.session_state.metadata is None:
        st.warning("No agent loaded. Launch with: `python -m adk_eval_tool <module> <variable>`")
        return

    tab_gen, tab_edit = st.tabs(["Generate", "View / Edit"])

    with tab_gen:
        st.subheader("Generate Tasks & Base Trajectories")

        constraints = st.text_area(
            "User constraints / context",
            placeholder="e.g., Focus on multi-tool tasks, include all sub-agents...",
            key="task_constraints",
        )
        model = st.selectbox("Generator model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])

        # Option to regenerate for specific tasks
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
                        merged_tasks = []
                        for task in existing.tasks:
                            if task.task_id in selected_tasks and task.task_id in new_map:
                                merged_tasks.append(new_map[task.task_id])
                            else:
                                merged_tasks.append(task)
                        existing_ids = {t.task_id for t in existing.tasks}
                        for task in result.tasks:
                            if task.task_id not in existing_ids:
                                merged_tasks.append(task)
                        existing.tasks = merged_tasks
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

    with tab_edit:
        if st.session_state.task_set is None:
            st.warning("Generate or load tasks first.")
            return

        task_set = st.session_state.task_set

        st.subheader(f"Agent: {task_set.agent_name} -- {len(task_set.tasks)} tasks")

        for i, task in enumerate(task_set.tasks):
            with st.expander(f"Task: {task.name} ({task.task_id}) -- {len(task.trajectories)} trajectories"):
                st.markdown(f"**Description:** {task.description}")

                edited = json_editor(task.model_dump(), key=f"task_{i}")
                if edited != task.model_dump():
                    try:
                        task_set.tasks[i] = Task.model_validate(edited)
                        st.session_state.task_set = task_set
                        st.success("Task updated.")
                    except Exception as e:
                        st.error(f"Invalid task data: {e}")

        st.divider()
        col_save, col_download = st.columns(2)
        with col_save:
            save_path = st.text_input("Save path", value=f"{task_set.agent_name}_tasks.json", key="task_save_path")
            if st.button("Save to disk", key="save_tasks"):
                Path(save_path).write_text(task_set.model_dump_json(indent=2))
                st.success(f"Saved to {save_path}")
        with col_download:
            st.download_button(
                "Download JSON",
                data=task_set.model_dump_json(indent=2),
                file_name=f"{task_set.agent_name}_tasks.json",
                mime="application/json",
            )
