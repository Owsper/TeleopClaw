import numpy as np
import pytest

from dreamhands.referee import CONTS,HMAX,Referee,parse
from dreamhands.sim import HOME,Sim

TASKS = ["put apple into the bowl","put orange into the bowl","put banana on the plate"]


@pytest.fixture(scope="module")
def sim():
	s = Sim(render=False)
	yield s
	s.close()


def settle(sim,sec=3):
	for i in range(int(sec * 20)):
		sim.step(HOME,HOME)


def test_parse():
	assert parse("put banana on the plate") == ("banana","plate")
	assert parse("Put the Orange into the bowl") == ("orange","bowl")
	assert parse("do a backflip") == ("apple","bowl")
	assert parse(None) == ("apple","bowl")


def test_sizes_match_model(sim):
	m,d = sim.model,sim.data
	sim.reset()
	g = m.body("plate").geomadr[0]
	assert m.geom_size[g][0] == pytest.approx(CONTS["plate"]["r"])
	assert m.geom_size[g][1] == pytest.approx(CONTS["plate"]["surf"])
	b = m.body("bowl")
	g = b.geomadr[0]
	mid = m.geom_dataid[g]
	v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
	w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3,3).T - d.xpos[b.id]
	rmax = np.hypot(w[:,0],w[:,1]).max()
	# outer rim 0.083 from build_xml, the inner rim the referee uses is a bit smaller
	assert rmax == pytest.approx(0.083,abs=1e-3)
	assert 0 < rmax - CONTS["bowl"]["r"] < 0.01


@pytest.mark.parametrize("task",TASKS)
def test_false_after_reset(sim,task):
	sim.reset()
	ref = Referee(sim,task)
	assert not ref.check()
	settle(sim,1)
	assert not ref()


@pytest.mark.parametrize("task",TASKS)
def test_true_when_dropped_in(sim,task):
	sim.reset()
	ref = Referee(sim,task)
	ref.drop()
	assert not ref.check()
	settle(sim)
	inf = ref.info()
	assert inf["ok"] and ref.check(),inf
	assert inf["touch"] and not inf["held"]


@pytest.mark.parametrize("task",TASKS)
def test_false_elsewhere_on_table(sim,task):
	sim.reset()
	ref = Referee(sim,task)
	ref.place([0.4,-0.4,0.82])
	settle(sim)
	inf = ref.info()
	assert not inf["ok"],inf
	assert inf["v"] < 0.05 and inf["dxy"] > 0.2


@pytest.mark.parametrize("task",TASKS)
def test_false_high_in_air(sim,task):
	sim.reset()
	ref = Referee(sim,task)
	ref.drop(0.3)
	inf = ref.info()
	assert not inf["ok"],inf
	assert inf["dxy"] < 0.01 and inf["v"] == 0 and inf["dz"] > 0.2


def test_false_hovering_just_over_the_floor(sim):
	# xy, height and speed all pass, but it floats 4 mm over the bowl floor : not lying in it
	sim.reset()
	ref = Referee(sim,"put apple into the bowl")
	ref.drop(0.045)
	inf = ref.info()
	assert inf["dxy"] < 0.01 and inf["v"] == 0 and 0 < inf["dz"] <= HMAX
	assert not inf["touch"] and not inf["ok"]
