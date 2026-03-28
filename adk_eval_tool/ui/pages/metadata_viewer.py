"""Page: View and edit agent metadata (loaded at launch)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import AgentMetadata
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Agent Metadata")

    if st.session_state.metadata is None:
        st.warning(
            "No agent metadata loaded. Launch the tool with an agent:\n\n"
            "```\npython -m adk_eval_tool <agent_module> <agent_variable>\n```"
        )
        return

    metadata = st.session_state.metadata

    # Agent tree visualization
    st.subheader("Agent Tree")
    _render_agent_tree(metadata)

    st.divider()

    # Editable JSON
    st.subheader("Edit Metadata")
    edited = json_editor(metadata.model_dump(), key="metadata")
    if edited != metadata.model_dump():
        try:
            st.session_state.metadata = AgentMetadata.model_validate(edited)
            st.success("Metadata updated.")
        except Exception as e:
            st.error(f"Invalid metadata: {e}")

    # Save / download
    col_save, col_download = st.columns(2)
    with col_save:
        save_path = st.text_input("Save path", value=f"{metadata.name}_metadata.json")
        if st.button("Save to disk"):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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
