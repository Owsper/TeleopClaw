import numpy as np

from dreamhands.runner import TeleopRunner
from dreamhands.scene import HI,LO,scatter
from dreamhands.sim import Sim
from dreamhands.trials import Referee,scripted_pick


def test_scatter_stays_in_the_box():
	sim = Sim(render = False)
	p0 = sim.data.body("apple").xpos[:2].copy()
	for s in range(20):
		sim.reset()
		d = scatter(sim,s)
		assert np.all(d >= LO) and np.all(d <= HI)
	sim.step(np.r_[np.zeros(6),1.0],np.r_[np.zeros(6),1.0])
	assert np.allclose(sim.data.body("apple").xpos[:2] - p0,d,atol = 5e-3)
	assert not np.allclose(scatter(sim,1),scatter(sim,2))
	sim.close()


def test_pick_works_from_scattered_starts():
	r = TeleopRunner(Sim(render = False))
	ref = Referee(r.sim)
	for s in (0,1,2):
		r.sim.reset()
		scatter(r.sim,s)
		r.go_ready()
		ref.reset()
		assert scripted_pick(r,ref).success,s
	r.sim.close()
