import json

import numpy as np

from dreamhands.episode import CAMS,Episode
from dreamhands.quality import QCfg,Verdict,judge,log,summary
from dreamhands.types import HandState,Tick

IMG = np.zeros((2,2,3),np.uint8)


def vec():
	v = np.full(14,0.5,np.float32)
	v[[6,13]] = 1.0
	return v


def mk(ticks=100,lost=0,ik=0,**kw):
	ep = Episode(task="pick cube",**kw)
	for i in range(ticks):
		v = vec()
		ep.tstate.append(v)
		ep.taction.append(v.copy())
		ep.flags.append({"tracking_lost": i < lost,"ik_reject": i < ik,"workspace_edge": False,"step_ms": 3.1})
		ep.modes.append("ENGAGED")
		if i % 2 == 0:
			ep.state.append(v)
			ep.action.append(v.copy())
			for c in CAMS:
				ep.images[c].append(IMG)
	return ep


def test_clean_keeps():
	v = judge(mk())
	assert v.keep and v.reasons == [] and str(v) == "KEEP"
	assert v.task == "pick cube"
	assert v.stats == {"dur": 5.0,"frames": 50,"ticks": 100,"lost": 0.0,"ik": 0.0,"edge": 0.0,"jump": 0.0,"guard": 0,"success": None}


def test_stats_plain_numbers():
	ep = mk(lost=2)
	ep.success = np.True_
	st = judge(ep).stats
	json.dumps(st,allow_nan=False)
	assert all(type(x) in (int,float,bool) for x in st.values())


def test_through_add():
	ep = Episode(task="t")
	for i in range(100):
		img = {c: IMG for c in CAMS} if i % 2 == 0 else None
		fl = {"tracking_lost": False,"ik_reject": False,"workspace_edge": i < 10,"ik_err_mm": 0.4}
		ep.add(Tick(step=i,t=i * 0.05,state=vec(),action=vec(),images=img,hand=HandState(),mode="ENGAGED",flags=fl))
	v = judge(ep)
	assert v.keep, str(v)
	assert v.stats["frames"] == 50 and v.stats["ticks"] == 100
	assert v.stats["edge"] == 0.1  # info only, never a drop


def test_referee():
	ep = mk()
	ep.success = False
	assert judge(ep).reasons == ["referee : fail"]
	ep.success = np.False_
	assert judge(ep).reasons == ["referee : fail"]
	ep.success = True
	assert judge(ep,QCfg(need_success=True)).keep
	ep.success = None
	assert judge(ep).keep
	v = judge(ep,QCfg(need_success=True))
	assert not v.keep and v.reasons == ["referee : not judged"]


def test_length():
	assert judge(mk(34)).keep  # 17 frames = Dream's floor at 10 fps
	v = judge(mk(32))  # 16 frames = exactly 1.6 s : dur rule ok, frame floor not
	assert v.stats["dur"] == 1.6 and v.reasons == ["frames : 16 < 17"]
	assert judge(mk(30)).reasons == ["dur : 1.50 s < 1.6 s","frames : 15 < 17"]


def test_dur_boundary():
	cfg = QCfg(min_dur=2.0)
	assert judge(mk(40),cfg).keep  # exactly min_dur keeps
	assert judge(mk(38),cfg).reasons == ["dur : 1.90 s < 2 s"]


def test_frame_floor_follows_fps():
	assert judge(mk(98,fps=30)).keep  # 49 frames
	assert judge(mk(96,fps=30)).reasons == ["frames : 48 < 49"]


def test_track_loss():
	assert judge(mk(1000,lost=49)).keep  # 4.9 %
	assert judge(mk(1000,lost=50)).reasons == ["track loss : 5.0 % >= 5 %"]
	v = judge(mk(100,lost=8))
	assert not v.keep and v.reasons == ["track loss : 8.0 % >= 5 %"]
	assert v.stats["lost"] == 0.08


def test_ik_rejects():
	assert judge(mk(100,ik=2)).keep
	v = judge(mk(100,ik=3))
	assert v.reasons == ["ik reject : 3.0 % >= 3 %"] and v.stats["ik"] == 0.03


def jumped(key,idx,d,cfg=None):
	ep = mk()
	for a in getattr(ep,key)[50:]:
		a[idx] += d
	return judge(ep,cfg)


def test_joint_jump():
	v = jumped("tstate",2,0.3)
	assert v.reasons == ["joint jump : 0.300 rad > 0.25 rad"] and v.stats["jump"] == 0.3
	assert jumped("taction",9,-0.4).reasons == ["joint jump : 0.400 rad > 0.25 rad"]
	assert jumped("tstate",12,0.26).reasons == ["joint jump : 0.260 rad > 0.25 rad"]
	assert jumped("taction",0,0.25).keep  # at the limit is fine
	assert jumped("tstate",2,0.3,QCfg(max_jump=0.5)).keep


def test_gripper_snap_is_not_a_jump():
	ep = mk()
	for key in ("tstate","taction"):
		for a in getattr(ep,key)[30:60]:
			a[[6,13]] = 0.0
	v = judge(ep)
	assert v.keep and v.stats["jump"] == 0.0


def test_guard():
	ep = mk()
	ep.flags[3]["guard"] = False
	ep.flags[4]["guard_reason"] = ""
	assert judge(ep).keep
	ep.flags[10]["guard_stop"] = True
	ep.flags[11]["guard"] = "force limit"
	v = judge(ep)
	assert v.reasons == ["guard : 2 ticks"] and v.stats["guard"] == 2


def test_non_finite():
	for key,val in (("tstate",np.nan),("taction",np.inf)):
		ep = mk()
		getattr(ep,key)[20][3] = val
		v = judge(ep)
		assert v.reasons == ["non-finite"]
		json.dumps(v.stats,allow_nan=False)


def test_empty():
	v = judge(Episode())
	assert not v.keep and str(v) == "DROP : empty"
	ep = mk()
	ep.flags.clear()
	assert judge(ep).reasons == ["empty"]


def test_several_reasons():
	ep = mk(100,lost=8,ik=5)
	ep.success = False
	ep.flags[50]["guard_x"] = 1
	v = judge(ep)
	assert v.reasons == ["referee : fail","track loss : 8.0 % >= 5 %","ik reject : 5.0 % >= 3 %","guard : 1 ticks"]
	assert str(v) == "DROP : referee : fail; track loss : 8.0 % >= 5 %; ik reject : 5.0 % >= 3 %; guard : 1 ticks"


def test_log_summary(tmp_path):
	p = tmp_path / "sub" / "q.jsonl"
	assert summary(p) == {"total": 0,"kept": 0,"dropped": 0,"reasons": {}}
	log(p,judge(mk()),0)
	log(p,judge(mk(100,lost=8,ik=5)),np.int64(1))
	log(str(p),judge(mk(100,lost=10)))
	rows = [json.loads(l) for l in p.read_text().splitlines()]
	assert [r["idx"] for r in rows] == [0,1,None]
	assert all(set(r) == {"idx","keep","reasons","stats","task"} for r in rows)
	assert rows[0]["keep"] is True and rows[0]["reasons"] == [] and rows[0]["task"] == "pick cube"
	assert rows[1]["reasons"] == ["track loss : 8.0 % >= 5 %","ik reject : 5.0 % >= 3 %"]
	assert rows[1]["stats"]["lost"] == 0.08 and rows[1]["stats"]["frames"] == 50
	assert summary(p) == {"total": 3,"kept": 1,"dropped": 2,"reasons": {"track loss": 2,"ik reject": 1}}


def test_verdict_str():
	assert str(Verdict(True)) == "KEEP"
	assert str(Verdict(False,["a : 1","b"])) == "DROP : a : 1; b"
