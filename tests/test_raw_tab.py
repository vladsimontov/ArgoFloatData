"""Raw tab regression test: a float with fewer cycles than the default stride.

The "Every Nth cycle" number_input takes its max_value from the data
(the number of cycles the float has), so a hardcoded value=10 raises
StreamlitValueAboveMaxError before the widget ever draws, and the whole
Raw tab dies. 849 of 20347 floats have fewer than 10 profiles, so this is
a live crash, not a corner case. The fix clamps: value=min(10, _nth_max).

This drives the real app through AppTest: type a WMO in the sidebar search
(a typed WMO bypasses every filter), pick the float, and let the script run.
Streamlit executes every `with tab_*:` body on each run regardless of which
tab is focused, so the Raw tab really does render here.

Fixture float 4903932 (BGC, coriolis, 3 cycles) has the same shape as the
reported 4904029. Its Sprof.nc is not committed (argo_local/dac/** is
gitignored, the profiles are hundreds of GB at full scale), so this skips
when the data is not synced instead of failing a data-less CI run.
"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
ARGO_ROOT = REPO / "argo_local"
APP = REPO / "argo_dashboard.py"

# BGC float with only 3 cycles, i.e. fewer than the default stride of 10.
WMO = "4903932"
SPROF = ARGO_ROOT / "dac" / "coriolis" / WMO / f"{WMO}_Sprof.nc"
N_CYCLES = 3


def _load_raw_tab():
    """Boot the app, search for the fixture WMO, select it, return the AppTest."""
    os.environ["ARGO_ROOT"] = str(ARGO_ROOT)
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, f"App raised on boot: {[e.value for e in at.exception]}"

    # The sidebar search is one free-text box for a WMO or a serial number.
    at.text_input[0].set_value(WMO).run()
    assert not at.exception, f"App raised on search: {[e.value for e in at.exception]}"

    picker = [s for s in at.selectbox if s.key == "wmo_pick"]
    assert picker, "the WMO picker did not render"
    assert WMO in picker[0].options, (
        f"search for {WMO} did not surface it; options were {picker[0].options}")

    # Selecting the float loads its Sprof and runs every tab body, Raw included.
    picker[0].select(WMO).run()
    return at


@pytest.mark.skipif(not SPROF.exists(),
                    reason=f"{SPROF} not synced locally; run the sync to enable")
def test_raw_tab_survives_float_with_fewer_cycles_than_stride():
    at = _load_raw_tab()

    # The bug surfaced as a hard exception, so name it: a bare "not at.exception"
    # would not say which failure we are guarding against.
    messages = [e.value for e in at.exception]
    assert not any("greater than the `max_value`" in m for m in messages), (
        f"Raw tab raised StreamlitValueAboveMaxError for a {N_CYCLES}-cycle "
        f"float: {messages}")
    assert not at.exception, f"Raw tab raised: {messages}"


@pytest.mark.skipif(not SPROF.exists(),
                    reason=f"{SPROF} not synced locally; run the sync to enable")
def test_every_nth_widget_clamps_to_the_cycle_count():
    at = _load_raw_tab()

    # Positive proof the widget actually drew. On the buggy code it is absent,
    # because number_input raises before it renders, so this fails there too.
    nth = [n for n in at.number_input if n.label == "Every Nth cycle"]
    assert nth, ("the 'Every Nth cycle' widget did not render, so the Raw tab "
                 "never got that far")
    assert nth[0].value == min(10, N_CYCLES), (
        f"expected the default stride clamped to {min(10, N_CYCLES)} for a "
        f"{N_CYCLES}-cycle float, got {nth[0].value}")
