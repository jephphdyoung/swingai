"""Streamlit wrapper for the draggable/resizable ghost-overlay editor.

Static bidirectional component (no Node build): serves components/overlay_editor/
and receives the adjusted transform back from the browser.
"""

import os
import streamlit.components.v1 as components

_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "overlay_editor")

_overlay_editor = components.declare_component("overlay_editor", path=_BUILD_DIR)


def overlay_editor(pairs, transform, key=None):
    """Render the overlay editor.

    Args:
        pairs: list of {"p", "user", "ref", "ref_ar"} still-frame dicts.
        transform: initial (auto-registration) transform dict.
        key: Streamlit widget key.

    Returns:
        The current transform dict (the seed until the user adjusts it).
    """
    return _overlay_editor(pairs=pairs, transform=transform, default=transform, key=key)
