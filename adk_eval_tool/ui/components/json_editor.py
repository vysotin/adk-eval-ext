"""JSON tree editor widget for Streamlit."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def json_editor(
    data: dict[str, Any],
    key: str,
    readonly: bool = False,
) -> dict[str, Any]:
    """Render an editable JSON tree.

    Args:
        data: The JSON data to display/edit.
        key: Unique Streamlit key prefix.
        readonly: If True, display only.

    Returns:
        The edited (or original) data.
    """
    json_str = json.dumps(data, indent=2)

    if readonly:
        st.code(json_str, language="json")
        return data

    edited_str = st.text_area(
        "Edit JSON",
        value=json_str,
        height=400,
        key=f"{key}_editor",
    )

    try:
        edited_data = json.loads(edited_str)
        return edited_data
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return data
