"""Headless boot smoke test.

Runs the whole Streamlit script once with no interaction, using Streamlit's
AppTest harness. With no float selected the app stops before any GDAC fetch, so
this needs no network: it just proves the app imports and renders on this
Python + dependency set. In CI this runs on Linux + Python 3.12 against the
pinned requirements, i.e. the same environment Streamlit Community Cloud builds,
so a bad wheel, a missing/renamed Streamlit API (e.g. st.iframe), or an
import-time crash fails the build instead of the live deployment.
"""
from streamlit.testing.v1 import AppTest


def test_app_boots_without_exception():
    at = AppTest.from_file("argo_dashboard.py", default_timeout=180)
    at.run()
    # at.exception is an ElementList, empty (falsy) when the boot is clean.
    assert not at.exception, f"App raised on boot: {list(at.exception)}"


def test_search_ui_renders():
    # The sidebar search (serial + WMO text inputs) must render; this catches a
    # boot that silently produces nothing rather than erroring.
    at = AppTest.from_file("argo_dashboard.py", default_timeout=180)
    at.run()
    assert not at.exception, f"App raised on boot: {list(at.exception)}"
    assert len(at.text_input) >= 1, "expected the search text inputs to render"
