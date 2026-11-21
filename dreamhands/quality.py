"""KEEP / DROP gate for one recorded Episode. Verdicts go to a jsonl the report reads."""
import json
from dataclasses import dataclass,field
from pathlib import Path

import numpy as np

ARM = [0,1,2,3,4,5,7,8,9,10,11,12]  # grippers 6 and 13 snap 0<->1, not a jump


@dataclass
class QCfg:
	min_dur: float = 1.6
	max_lost: float = 0.05
	max_ik: float = 0.03
	# rad per 20 Hz tick : same as the IK gate (ik.py), anything bigger went around IK
	max_jump: float = 0.25
	need_success: bool = False


@dataclass
class Verdict:
	keep: bool
	reasons: list = field(default_factory=list)
	stats: dict = field(default_factory=dict)
	task: str = ""

	def __str__(self):
		return "KEEP" if self.keep else "DROP : " + "; ".join(self.reasons)


def share(flags,key):
	return sum(bool(f.get(key)) for f in flags) / max(len(flags),1)


def maxJump(seq):
	d = np.abs(np.diff(np.asarray(seq,float).reshape(-1,14)[:,ARM],axis=0))
	d = d[np.isfinite(d)]
	return float(d.max()) if d.size else 0.0


def judge(ep,cfg=None):
	cfg = cfg or QCfg()
	ticks = len(ep.flags)
	minf = 48 * ep.fps // 30 + 1  # Dream's validator refuses shorter
	lost = share(ep.flags,"tracking_lost")
	ik = share(ep.flags,"ik_reject")
	jump = max(maxJump(ep.tstate),maxJump(ep.taction))
	guard = sum(any(v for k,v in f.items() if k.startswith("guard")) for f in ep.flags)
	fin = all(np.isfinite(np.asarray(a,float)).all() for a in (ep.tstate,ep.taction,ep.state,ep.action))
	ok = None if ep.success is None else bool(ep.success)

	why = []
	if not ep.n or not ticks:
		why.append("empty")
	else:
		if ok is False:
			why.append("referee : fail")
		elif ok is None and cfg.need_success:
			why.append("referee : not judged")
		if ep.dur < cfg.min_dur:
			why.append(f"dur : {ep.dur:.2f} s < {cfg.min_dur:g} s")
		if ep.n < minf:
			why.append(f"frames : {ep.n} < {minf}")
		if lost >= cfg.max_lost:
			why.append(f"track loss : {lost * 100:.1f} % >= {cfg.max_lost * 100:g} %")
		if ik >= cfg.max_ik:
			why.append(f"ik reject : {ik * 100:.1f} % >= {cfg.max_ik * 100:g} %")
		if jump > cfg.max_jump + 1e-6:  # float32 storage noise
			why.append(f"joint jump : {jump:.3f} rad > {cfg.max_jump:g} rad")
		if guard:
			why.append(f"guard : {guard} ticks")
		if not fin:
			why.append("non-finite")

	stats = {
		"dur": round(ep.dur,2),
		"frames": ep.n,
		"ticks": ticks,
		"lost": round(lost,4),
		"ik": round(ik,4),
		"edge": round(share(ep.flags,"workspace_edge"),4),
		"jump": round(jump,4),
		"guard": guard,
		"success": ok,
	}
	return Verdict(not why,why,stats,ep.task)


def log(path,v,idx=None):
	p = Path(path)
	p.parent.mkdir(parents=True,exist_ok=True)
	row = {"idx": idx if idx is None else int(idx),"keep": v.keep,"reasons": v.reasons,"stats": v.stats,"task": v.task}
	with p.open("a") as f:
		f.write(json.dumps(row) + "\n")


def summary(path):
	p = Path(path)
	rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
	kept = sum(r["keep"] for r in rows)
	why = {}
	for r in rows:
		for s in r["reasons"]:
			k = s.split(" : ")[0]
			why[k] = why.get(k,0) + 1
	return {"total": len(rows),"kept": kept,"dropped": len(rows) - kept,"reasons": why}
