import numpy as np

from dreamhands.runner import TeleopRunner
from dreamhands.sim import Sim
from dreamhands.trials import Referee, scripted_pick
from dreamhands.types import HandState


def test_scripted_pick_lifts_apple():
    """The grasp is physically feasible through our IK/sim path (go/no-go baseline)."""
    r = TeleopRunner(Sim(render=False))
    r.go_ready()
    v = scripted_pick(r, Referee(r.sim))
    assert v.success, v
    r.sim.close()


def test_referee_does_not_pass_untouched_apple():
    r = TeleopRunner(Sim(render=False))
    r.go_ready()
    ref = Referee(r.sim)
    for i in range(40):
        hand = HandState(t=float(i), present=True, confidence=1.0, pos=np.full(3, 0.5))
        v = ref(r.tick(hand, now=float(i)))
    assert not v.done and abs(v.lift) < 0.005
    r.sim.close()
