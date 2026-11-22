import time
from types import SimpleNamespace as NS

import numpy as np

from dreamhands import ui
from dreamhands.types import HandState, Tick


def _tick(present=True, mode="ENGAGED"):
    hand = HandState(t=time.monotonic() - 0.058, present=present, pinch=0.42)
    return Tick(step=0, t=0.0, state=np.zeros(14), action=np.zeros(14), images=None,
                hand=hand, mode=mode, flags={"tracking_lost": not present})


def test_sim_pane_keeps_its_place_and_webcam_is_scaled_beside_it():
    sim = np.full((360, 640, 3), 7, np.uint8)
    cam = np.full((480, 640, 3), 200, np.uint8)
    out = ui.compose(sim, cam, [("TRACKING OK", ui.GREEN)])
    assert out.shape == (360 + ui.BAR_H, 640 + 480, 3)
    assert (out[:360, :640] == 7).all()  # mouse coordinates in the sim pane are unchanged
    assert (out[:360, 640:] == 200).all()


def test_mouse_mode_has_no_webcam_pane():
    out = ui.compose(np.zeros((360, 640, 3), np.uint8), None, [])
    assert out.shape == (360 + ui.BAR_H, 640, 3)


def test_status_items_follow_the_spec():
    rec = NS(ep=NS(n=487), done=15, on_done=NS(w=NS(n=12)))
    texts = [t for t, _ in ui.status(_tick(), None, rec)]
    assert texts[:3] == ["TRACKING OK", "ENGAGED OK", "REC"]
    assert "pinch 0.42" in texts and "frames 487" in texts and "kept 12/15" in texts
    assert any(t.startswith("latency ") and t.endswith(" ms") for t in texts)
    lost = [t for t, _ in ui.status(_tick(present=False, mode="HOLD"), None)]
    assert lost[:2] == ["TRACKING --", "ENGAGED --"] and not any(t.startswith("latency") for t in lost)
