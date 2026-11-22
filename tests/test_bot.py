import numpy as np

from dreamhands import bot
from dreamhands.runner import TeleopRunner
from dreamhands.sim import Sim
from dreamhands.trials import Referee


def test_pick():
	r = TeleopRunner(Sim(render = False))
	ref = Referee(r.sim)
	got = []
	r.on_tick.append(got.append)
	rng = np.random.default_rng(7)
	hand = bot.Hand(r,rng)
	assert sum(bool(bot.episode(r,ref,hand,rng).success) for i in range(3)) >= 2
	# it went through the teleop path : engaged, precision near the apple, a clutch, no bad IK
	assert {t.mode for t in got} >= {"ENGAGED","READY"}
	assert any(t.flags.get("precision") for t in got)
	assert any(t.hand.fist for t in got)
	assert not any(t.flags.get("ik_reject") for t in got)
