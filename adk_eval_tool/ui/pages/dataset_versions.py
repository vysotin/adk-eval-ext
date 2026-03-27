"""Page: Manage versions of generated datasets."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from adk_eval_tool.ui.components.json_editor import json_editor


VERSIONS_DIR = "eval_versions"


def render():
    st.header("Dataset Versions")

    tab_create, tab_browse = st.tabs(["Create Version", "Browse Versions"])

    with tab_create:
        _render_create_version()

    with tab_browse:
        _render_browse_versions()


def _render_create_version():
    st.subheader("Create New Version")

    if not st.session_state.eval_sets and not st.session_state.intent_set:
        st.warning("No data to version. Generate metadata, intents, or test cases first.")
        return

    version_name = st.text_input(
        "Version name",
        value=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    version_notes = st.text_area("Version notes", placeholder="What changed in this version...")

    source_dir = st.text_input("Source eval_datasets directory", value="eval_datasets")

    if st.button("Create Version"):
        version_path = Path(VERSIONS_DIR) / version_name
        version_path.mkdir(parents=True, exist_ok=True)

        if st.session_state.metadata:
            (version_path / "metadata.json").write_text(
                st.session_state.metadata.model_dump_json(indent=2)
            )

        if st.session_state.intent_set:
            (version_path / "intents.json").write_text(
                st.session_state.intent_set.model_dump_json(indent=2)
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
                (evalsets_dir / f"{eid}.evalset.json").write_text(
                    json.dumps(es, indent=2)
                )

        manifest = {
            "version": version_name,
            "created_at": datetime.now().isoformat(),
            "notes": version_notes,
            "agent_name": st.session_state.metadata.name if st.session_state.metadata else None,
            "num_intents": len(st.session_state.intent_set.intents) if st.session_state.intent_set else 0,
            "num_eval_sets": len(st.session_state.eval_sets),
        }
        (version_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

        st.success(f"Version '{version_name}' created at {version_path}")


def _render_browse_versions():
    versions_path = Path(VERSIONS_DIR)
    if not versions_path.exists():
        st.info("No versions created yet.")
        return

    versions = sorted(versions_path.iterdir(), reverse=True)
    if not versions:
        st.info("No versions found.")
        return

    for version_dir in versions:
        if not version_dir.is_dir():
            continue

        manifest_file = version_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
        else:
            manifest = {"version": version_dir.name}

        with st.expander(
            f"v{manifest.get('version', version_dir.name)} -- "
            f"{manifest.get('created_at', 'unknown')} -- "
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
                if st.button(f"Load version", key=f"load_{version_dir.name}"):
                    _load_version(version_dir)
                    st.success(f"Loaded version {version_dir.name}")
                    st.rerun()
            with col_delete:
                if st.button(f"Delete version", key=f"del_{version_dir.name}", type="secondary"):
                    shutil.rmtree(version_dir)
                    st.success(f"Deleted version {version_dir.name}")
                    st.rerun()


def _load_version(version_dir: Path):
    """Load a version's data into session state."""
    from adk_eval_tool.schemas import AgentMetadata, IntentScenarioSet

    meta_file = version_dir / "metadata.json"
    if meta_file.exists():
        st.session_state.metadata = AgentMetadata.model_validate(
            json.loads(meta_file.read_text())
        )

    intents_file = version_dir / "intents.json"
    if intents_file.exists():
        st.session_state.intent_set = IntentScenarioSet.model_validate(
            json.loads(intents_file.read_text())
        )

    evalsets_dir = version_dir / "evalsets"
    if evalsets_dir.exists():
        st.session_state.eval_sets = [
            json.loads(f.read_text())
            for f in sorted(evalsets_dir.glob("*.evalset.json"))
        ]
