"""Page: Generate, view, and edit intents & scenarios."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from adk_eval_tool.schemas import IntentScenarioSet, Intent
from adk_eval_tool.ui.components.json_editor import json_editor


def render():
    st.header("Intents & Scenarios")

    if st.session_state.metadata is None:
        st.warning("Load agent metadata first (Agent Metadata page).")
        return

    tab_gen, tab_edit = st.tabs(["Generate", "View / Edit"])

    with tab_gen:
        st.subheader("Generate Intents & Scenarios")

        constraints = st.text_area(
            "User constraints / context",
            placeholder="e.g., Focus on error handling scenarios, include multi-language inputs...",
            key="intent_constraints",
        )
        num_scenarios = st.slider("Scenarios per intent", min_value=1, max_value=10, value=3)
        model = st.selectbox("Generator model", ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"])

        if st.session_state.intent_set and st.session_state.intent_set.intents:
            st.divider()
            st.markdown("**Selective regeneration:**")
            existing_intents = [i.intent_id for i in st.session_state.intent_set.intents]
            selected_intents = st.multiselect(
                "Regenerate only these intents (leave empty for full generation)",
                options=existing_intents,
            )
        else:
            selected_intents = []

        if st.button("Generate", key="gen_intents"):
            with st.spinner("Generating intents and scenarios..."):
                from adk_eval_tool.intent_generator import generate_intents

                try:
                    result = asyncio.run(generate_intents(
                        metadata=st.session_state.metadata,
                        user_constraints=constraints,
                        num_scenarios_per_intent=num_scenarios,
                        model=model,
                    ))

                    if selected_intents and st.session_state.intent_set:
                        existing = st.session_state.intent_set
                        new_map = {i.intent_id: i for i in result.intents}
                        merged_intents = []
                        for intent in existing.intents:
                            if intent.intent_id in selected_intents and intent.intent_id in new_map:
                                merged_intents.append(new_map[intent.intent_id])
                            else:
                                merged_intents.append(intent)
                        existing_ids = {i.intent_id for i in existing.intents}
                        for intent in result.intents:
                            if intent.intent_id not in existing_ids:
                                merged_intents.append(intent)
                        existing.intents = merged_intents
                        st.session_state.intent_set = existing
                    else:
                        st.session_state.intent_set = result

                    st.success(f"Generated {len(st.session_state.intent_set.intents)} intents")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        st.divider()
        uploaded = st.file_uploader("Or upload existing intents JSON", type=["json"], key="intent_upload")
        if uploaded:
            try:
                data = json.loads(uploaded.read())
                st.session_state.intent_set = IntentScenarioSet.model_validate(data)
                st.success(f"Loaded {len(st.session_state.intent_set.intents)} intents")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    with tab_edit:
        if st.session_state.intent_set is None:
            st.warning("Generate or load intents first.")
            return

        intent_set = st.session_state.intent_set

        st.subheader(f"Agent: {intent_set.agent_name} -- {len(intent_set.intents)} intents")

        for i, intent in enumerate(intent_set.intents):
            with st.expander(f"Intent: {intent.name} ({intent.intent_id}) -- {len(intent.scenarios)} scenarios"):
                st.markdown(f"**Category:** {intent.category}")
                st.markdown(f"**Description:** {intent.description}")

                edited = json_editor(intent.model_dump(), key=f"intent_{i}")
                if edited != intent.model_dump():
                    try:
                        intent_set.intents[i] = Intent.model_validate(edited)
                        st.session_state.intent_set = intent_set
                        st.success("Intent updated.")
                    except Exception as e:
                        st.error(f"Invalid intent data: {e}")

        st.divider()
        col_save, col_download = st.columns(2)
        with col_save:
            save_path = st.text_input("Save path", value=f"{intent_set.agent_name}_intents.json", key="intent_save_path")
            if st.button("Save to disk", key="save_intents"):
                Path(save_path).write_text(intent_set.model_dump_json(indent=2))
                st.success(f"Saved to {save_path}")
        with col_download:
            st.download_button(
                "Download JSON",
                data=intent_set.model_dump_json(indent=2),
                file_name=f"{intent_set.agent_name}_intents.json",
                mime="application/json",
            )
