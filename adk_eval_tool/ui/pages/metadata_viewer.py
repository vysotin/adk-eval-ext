"""Page: Parse agent, view, and edit metadata."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import AgentMetadata
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Agent Metadata")

    tab_load, tab_edit = st.tabs(["Load / Parse", "View / Edit"])

    with tab_load:
        st.subheader("Load from file")
        uploaded = st.file_uploader("Upload metadata JSON", type=["json"], key="meta_upload")
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                st.session_state.metadata = AgentMetadata.model_validate(data)
                st.success(f"Loaded metadata for agent: {st.session_state.metadata.name}")
            except Exception as e:
                st.error(f"Failed to load: {e}")

        st.divider()
        st.subheader("Parse from agent module")
        st.markdown(
            "To parse a live agent, provide the module path (e.g., `my_agent.agent`). "
            "The module must define a `root_agent` or named agent variable."
        )

        col1, col2 = st.columns(2)
        with col1:
            module_path = st.text_input("Agent module path", placeholder="my_agent.agent")
        with col2:
            agent_var = st.text_input("Agent variable name", value="root_agent")

        if st.button("Parse Agent", disabled=not module_path):
            try:
                import importlib
                mod = importlib.import_module(module_path)
                agent_obj = getattr(mod, agent_var)

                from adk_eval_tool.agent_parser import parse_agent
                st.session_state.metadata = parse_agent(agent_obj)
                st.success(f"Parsed agent: {st.session_state.metadata.name}")
            except Exception as e:
                st.error(f"Failed to parse: {e}")

    with tab_edit:
        if st.session_state.metadata is None:
            st.warning("No metadata loaded. Use the 'Load / Parse' tab first.")
            return

        metadata = st.session_state.metadata

        st.subheader("Agent Tree")
        _render_agent_tree(metadata)

        st.divider()

        st.subheader("Edit Metadata")
        edited = json_editor(metadata.model_dump(), key="metadata")
        if edited != metadata.model_dump():
            try:
                st.session_state.metadata = AgentMetadata.model_validate(edited)
                st.success("Metadata updated.")
            except Exception as e:
                st.error(f"Invalid metadata: {e}")

        col_save, col_download = st.columns(2)
        with col_save:
            save_path = st.text_input("Save path", value=f"{metadata.name}_metadata.json")
            if st.button("Save to disk"):
                Path(save_path).write_text(
                    st.session_state.metadata.model_dump_json(indent=2)
                )
                st.success(f"Saved to {save_path}")
        with col_download:
            st.download_button(
                "Download JSON",
                data=st.session_state.metadata.model_dump_json(indent=2),
                file_name=f"{metadata.name}_metadata.json",
                mime="application/json",
            )


def _render_agent_tree(metadata: AgentMetadata, level: int = 0):
    """Render agent tree using Streamlit expanders."""
    icon = "\U0001f4e6" if level == 0 else "\U0001f4ce"
    with st.expander(f"{icon} {metadata.name} ({metadata.agent_type})", expanded=(level == 0)):
        st.markdown(f"**Description:** {metadata.description}")
        st.markdown(f"**Model:** {metadata.model}")
        if metadata.instruction:
            st.markdown(f"**Instruction:** {metadata.instruction[:300]}{'...' if len(metadata.instruction) > 300 else ''}")
        if metadata.tools:
            st.markdown("**Tools:**")
            for tool in metadata.tools:
                st.markdown(f"- `{tool.name}` ({tool.source}): {tool.description}")
        if metadata.sub_agents:
            st.markdown("**Sub-agents:**")
            for sub in metadata.sub_agents:
                _render_agent_tree(sub, level + 1)
