import numpy as np

from dreamhands.episode import CAMS
from dreamhands.fake_sim import FakeSim
from dreamhands.recorder import Recorder

FULL = {0: ["record"],99: ["next"]}


def record(sim,**kw):
	eps,ticks = [],[]
	rec = Recorder("task",on_done=eps.append,**kw)
	sim.run(ticks.append)
	for tick in ticks:
		rec.on_tick(tick)
	return rec,eps,ticks


def test_fake_sim_ticks():
	ticks = list(FakeSim(5).ticks())
	assert len(ticks) == 100
	for i,tk in enumerate(ticks):
		assert tk.step == i and abs(tk.t - i * 0.05) < 1e-9
		assert tk.state.shape == tk.action.shape == (14,)
		assert tk.mode == "ENGAGED" and tk.events == []
		assert tk.flags == {"tracking_lost": False,"ik_reject": False,"workspace_edge": False}
		assert (tk.images is None) == (i % 2 == 1)
	grip = np.array([np.r_[tk.state[[6,13]],tk.action[[6,13]]] for tk in ticks])
	assert grip.min() >= 0 and grip.max() <= 1
	# smooth, action = next state
	st = np.array([tk.state for tk in ticks])
	assert np.abs(np.diff(st,axis=0)).max() < 0.05
	assert np.allclose(ticks[3].action,ticks[4].state)
	im = ticks[0].images
	assert set(im) == set(CAMS)
	for c in CAMS:
		assert im[c].shape == (360,640,3) and im[c].dtype == np.uint8
	assert not np.array_equal(im["top"],im["left"]) and not np.array_equal(im["left"],im["right"])
	assert not np.array_equal(im["top"],ticks[2].images["top"])
	assert np.array_equal(ticks[7].state,list(FakeSim(5).ticks())[7].state)


def test_5s_episode():
	rec,eps,ticks = record(FakeSim(5,events=FULL))
	assert len(eps) == 1 and rec.done == 1 and rec.dropped == 0 and rec.ep is None
	ep = eps[0]
	assert ep.task == "task" and ep.n == 50 and ep.fps == 10 and ep.dur == 5
	assert len(ep.tstate) == len(ep.taction) == len(ep.flags) == len(ep.modes) == 100
	assert np.array(ep.state).shape == np.array(ep.action).shape == (50,14)
	assert set(ep.images) == set(CAMS)
	for c in CAMS:
		assert len(ep.images[c]) == 50
		assert ep.images[c][0].shape == (360,640,3) and ep.images[c][0].dtype == np.uint8
	assert ep.success is None


def test_frames_from_even_steps_action_from_next_tick():
	rec,eps,ticks = record(FakeSim(5,events=FULL))
	ep = eps[0]
	for k in range(ep.n):
		assert np.allclose(ep.state[k],ticks[2 * k].state)
		assert ep.images["top"][k] is ticks[2 * k].images["top"]
		assert np.allclose(ep.action[k],ticks[2 * k + 1].action)
	assert not np.allclose(ep.action[0],ticks[0].action)


def test_events():
	ev = {10: ["record"],29: ["next"],40: ["record"],50: ["discard"],60: ["next"]}
	rec,eps,ticks = record(FakeSim(5,events=ev))
	assert len(eps) == 1 and rec.done == 1 and rec.dropped == 1 and rec.ep is None
	ep = eps[0]
	assert len(ep.tstate) == 20 and ep.n == 10
	assert np.allclose(ep.tstate[0],ticks[10].state)
	assert np.allclose(ep.tstate[-1],ticks[29].state)


def test_record_restarts():
	rec,eps,ticks = record(FakeSim(5,events={0: ["record"],20: ["record"],39: ["next"]}))
	assert len(eps) == 1 and rec.dropped == 1
	assert len(eps[0].tstate) == 20
	assert np.allclose(eps[0].tstate[0],ticks[20].state)


def test_rec_state_for_hud():
	rec = Recorder("task")
	assert rec.ep is None
	for tk in FakeSim(1,events={4: ["record"]}).ticks():
		rec.on_tick(tk)
		assert (rec.ep is None) == (tk.step < 4)
	assert len(rec.ep.tstate) == 16


def test_max_s():
	rec,eps,ticks = record(FakeSim(5,events={0: ["record"]}),max_s=1)
	assert len(eps) == 1 and rec.ep is None
	assert len(eps[0].tstate) == 20 and eps[0].n == 10


def test_referee():
	for ref,want in ((lambda: 1,True),(lambda: 0,False),(None,None)):
		rec,eps,ticks = record(FakeSim(1,events={0: ["record"],19: ["next"]}),referee=ref)
		assert eps[0].success is want


def test_start_stop():
	eps = []
	rec = Recorder("task",on_done=eps.append)
	assert rec.stop() is None
	rec.start()
	FakeSim(1).run(rec.on_tick)
	ep = rec.stop()
	assert eps == [ep] and ep.n == 10 and rec.ep is None and rec.done == 1
	rec.start()
	FakeSim(1).run(rec.on_tick)
	assert rec.stop(keep=False) is None
	assert len(eps) == 1 and rec.dropped == 1 and rec.ep is None


def test_faults():
	rec,eps,ticks = record(FakeSim(5,lost=0.1,ik=0.05,jump_at=50,events=FULL))
	ep = eps[0]
	assert sum(f["tracking_lost"] for f in ep.flags) == 10
	assert sum(f["ik_reject"] for f in ep.flags) == 5
	d = np.abs(np.diff(np.array(ep.tstate),axis=0)).max(1)
	assert d.argmax() == 49 and d[49] > 0.5
	assert np.delete(d,49).max() < 0.05
	assert [tk.step for tk in ticks if tk.events] == [0,99]
