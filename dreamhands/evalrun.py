"""Headless batch eval of one checkpoint in the sim, one jsonl line per episode.

python -m dreamhands.evalrun --model <checkpoint id> --task "pick up the apple" --episodes 10 --out data/eval.jsonl --tag ft
"""
import argparse
import json
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from . import sim as S
from . import trials
from .ik import TOP_DOWN,ArmIK
from .referee import Referee,parse
from .runner import READY_TCP
from .scene import scatter

DT = 0.05
RETRIES = 8
NET = ("DreamTimeout","DreamProtocolError")


def loadDc():
	if S.dc is not None:
		return S.dc
	d = str(Path(S._yam.__file__).parent)
	if d not in sys.path:
		sys.path.insert(0,d)
	import dream_control
	return dream_control


dc = loadDc()


class Place:
	# the geometric place detector (referee.py) behind the trials.Referee interface
	def __init__(self,sim,task,hold):
		self.ref,self.hold = Referee(sim,task),hold
		self.reset()

	def reset(self):
		self.held = 0
		self.verdict = NS(done=False,success=False,reason="")

	def __call__(self,tick):
		v = self.verdict
		self.held = self.held + DT if self.ref.check() else 0
		if self.held >= self.hold - 1e-9:
			v.done,v.success,v.reason = True,True,f"{self.ref.obj} placed"
		return v


def pick(sim,body):
	# right arm rows of the trials.scripted_pick path, solved on the IK's private data
	ik = ArmIK(sim.model)
	q = sim.state()["right"][:6]
	ready = ik.solve_global(READY_TCP,TOP_DOWN).q
	rows = [np.r_[q + (ready - q) * (0.5 - 0.5 * np.cos(np.pi * i / 40)),1.0] for i in range(1,41)]
	rows += [rows[-1]] * 10
	q,cur = ready,READY_TCP
	xy = sim.data.body(body).xpos[:2]
	up,down = trials.APPROACH_Z,trials.GRASP_Z
	for z,grip,n in ((up,1.0,40),(down,1.0,30),(down,0.0,15),(up,0.0,30),(up,0.0,40)):
		p = np.r_[xy,z]
		for i in range(1,n + 1):
			res,rej = ik.step(cur + (p - cur) * min(1,i / max(1,n - 5)),TOP_DOWN,q)
			if not rej:
				q = res.q
			rows.append(np.r_[q,grip])
		cur = p
	return rows


class Warming(RuntimeError):
	reason,retryable,retry_after_s = "pool_warming",True,1.0


class Fake:
	# dry-run stand-in for the dream client : one fake warm-up, idle on even episodes, scripted success on odd ones
	def __init__(self,run):
		self.run = run
		self.ep = -1
		self.cold = True

	def reset_episode(self):
		r = self.run
		self.ep += 1
		self.calls = 0
		self.script = self.ep % 2 == 1
		self.last = r.sim.state()["right"]
		self.rows = pick(r.sim,r.obj) if self.script and r.a.referee == "lift" else []

	def set_observation(self,frames,obs,timestamp=None):
		pass

	def get_action(self):
		if self.cold:
			self.cold = False
			raise Warming("model is starting")
		r = self.run
		self.calls += 1
		if self.script and r.a.referee == "place" and self.calls == 2:
			r.ref.ref.drop()
		rows,self.rows = self.rows[:48],self.rows[48:]
		if rows:
			self.last = rows[-1]
		rows = rows + [self.last] * (48 - len(rows))
		left = r.sim.state()["left"].tolist()
		return {"q_r_chunk": [x.tolist() for x in rows],"q_l_chunk": [left] * 48}

	def close(self):
		pass


def chunks(rep):
	# a fine-tune may answer only per_key action.<part> rows : build the YAM chunks from them
	if "q_r_chunk" in rep or not isinstance(rep.get("per_key"),dict):
		return rep
	pk,rep = rep["per_key"],dict(rep)
	for side,key in (("right","q_r_chunk"),("left","q_l_chunk")):
		arm = np.asarray(pk[f"action.{side}_arm"],float).reshape(-1,6)
		rep[key] = np.hstack([arm,np.asarray(pk[f"action.{side}_gripper"],float).reshape(-1,1)])
	return rep


class Run:
	def __init__(self,a,key=""):
		self.a,self.key = a,key
		self.sim = S.Sim(render=False)
		self.env = self.sim.env	# the dc helpers want the YamEnv itself
		self.obj = parse(a.task)[0]
		self.ref = trials.Referee(self.sim,self.obj,DT) if a.referee == "lift" else Place(self.sim,a.task,a.hold)
		self.poses = a.model in dc.POSE_MODELS
		self.pol = None
		self.fatal = False
		self.cam = True
		if a.dry_run:
			try:
				self.env.images()
			except Exception as e:
				print(f"no rendering ({type(e).__name__}) : dry-run without camera frames",flush=True)
				self.cam = False

	def open(self):
		if self.a.dry_run:
			return Fake(self)
		return dc.open_policy(dc.policy_config(self.a.model,None,self.poses),self.key)

	def feed(self,every):
		if self.cam:
			dc.feed(self.pol,self.env,self.win,self.a.task,self.a.model,self.steps,every=every,station=self.poses)

	def step(self,l,r,live=False):
		self.env.step(l,r)
		# the example renders every step for its viewer, the model only ever sees every 4th
		if self.cam and self.steps % dc.FRAME_EVERY_STEPS == 0:
			dc.capture(self.env,self.win)
			if live:
				self.feed(1)
		self.steps += 1
		return self.ref(NS(action=np.r_[r,l]))

	def wait(self,sec,why):
		if time.monotonic() - self.said > 20:
			self.said = time.monotonic()
			print(f"  {why} : waited {self.waited / 60:.1f} min, next try in {sec:.0f} s",flush=True)
		time.sleep(sec)
		self.waited += sec

	def ask(self):
		tries = restarts = 0
		while True:
			t = time.monotonic()
			try:
				rep = self.pol.get_action()
				self.rtt.append(time.monotonic() - t)
				return rep
			except Exception as e:
				name,reason = type(e).__name__,getattr(e,"reason",None)
				after = getattr(e,"retry_after_s",None)
				if name == "DreamSessionExpired" and restarts < 3:
					restarts += 1
					print("  the model restarted the session : asking again",flush=True)
				elif not self.poses and dc.needs_poses(e):
					print("  this model reads hand poses : reopening with the YAM station keys",flush=True)
					self.pol.close()
					self.poses = True
					self.pol = self.open()
					self.feed(0)
				elif reason == "pool_warming" and self.waited < self.a.warm_min * 60:
					self.wait(after or 15,"pool_warming (model is starting)")
				elif (getattr(e,"retryable",False) or after is not None or name in NET) and tries < RETRIES:
					tries += 1
					self.wait(min(60,max(after or 0,2 ** tries)),f"{reason or name} (try {tries}/{RETRIES})")
				else:
					raise

	def episode(self,ep):
		a,sim,env = self.a,self.sim,self.env
		sim.reset()
		if a.scatter:
			scatter(sim,ep,self.obj)  # seeded by ep, so every checkpoint sees the same starts
			sim.step(S.HOME,S.HOME)
		self.ref.reset()
		self.win = {n: deque(maxlen=dc.FRAMES) for n in dc.VIDEO_ROLES}
		self.steps = 0
		self.rtt = []
		self.waited = self.said = 0
		v,err = self.ref.verdict,None
		nmax = round(a.seconds / DT)
		t0 = time.monotonic()
		try:
			if self.pol is None:
				self.pol = self.open()
			self.pol.reset_episode()
			while self.cam and len(self.win["top"]) < dc.FRAMES:
				st = env.state()
				self.step(st["left"],st["right"])
			self.feed(0)
			while not v.done and self.steps < nmax and not err:
				rep = self.ask()
				try:
					left,right = dc.validate_chunk(env,chunks(rep),a.actions,verbose=False)
				except (ValueError,KeyError,TypeError) as e:
					err = f"bad chunk : {type(e).__name__} : {e}"
					break
				for l,r in zip(left,right):
					v = self.step(l,r,True)
					if v.done or self.steps >= nmax:
						break
		except Exception as e:
			err = f"{type(e).__name__} : {e}"
			self.fatal = True
		rtt = round(statistics.median(self.rtt) * 1000) if self.rtt else None
		return {
			"model": a.model,
			"tag": a.tag,
			"task": a.task,
			"ep": ep,
			"success": bool(v.success),
			"sim_s": round(self.steps * DT,2),
			"wall_s": round(time.monotonic() - t0 - self.waited,2),
			"calls": len(self.rtt),
			"rtt_ms": rtt,
			"error": err,
			"reason": "error" if err else v.reason or "timeout",
			"referee": a.referee,
		}

	def close(self):
		if self.pol is not None:
			try:
				self.pol.close()
			except Exception as e:
				print(f"session release failed, it will expire on the server : {e}")
		self.env.close()


def rows(path):
	p = Path(path)
	if not p.exists():
		return []
	out = []
	for line in p.read_text().splitlines():
		try:
			out.append(json.loads(line))
		except ValueError:
			pass
	return out


def mine(a,path):
	return [
		r for r in rows(path)
		if not r.get("error") and (r.get("model"),r.get("task"),r.get("tag"),r.get("referee","lift")) == (a.model,a.task,a.tag,a.referee)
	]


def main(argv=None):
	p = argparse.ArgumentParser(description="batch eval of one checkpoint in the sim")
	p.add_argument("--model",type=dc.resolve_model,default=dc.BETA_MODEL,help="beta, or a checkpoint / fine-tune id")
	p.add_argument("--task",default="pick up the apple")
	p.add_argument("--episodes",type=int,default=10)
	p.add_argument("--seconds",type=float,default=60,help="sim time budget per episode")
	p.add_argument("--out",default="data/eval.jsonl")
	p.add_argument("--tag",help="base | ft, the group name in the report")
	p.add_argument("--referee",choices=("lift","place"),default="lift",help="lift : team rule (trials.py), place : object in bowl / on plate")
	p.add_argument("--hold",type=float,default=0.5,help="place : sim s the object must stay placed")
	p.add_argument("--actions",type=int,default=48,help="rows executed per response, 1-48")
	p.add_argument("--scatter",action="store_true",help="randomised object start like the recordings (scene.py), seeded by episode")
	p.add_argument("--warm-min",type=float,default=90,help="max minutes to wait for a starting model")
	p.add_argument("--dry-run",action="store_true",help="no network : idle / scripted stand-in policy")
	a = p.parse_args(argv)
	if not 1 <= a.actions <= 48 or a.seconds <= 0 or not a.task.strip():
		p.error("use 1-48 actions, positive seconds and a non-empty task")

	key = ""
	if not a.dry_run:
		key = dc.settings().get("DREAM_API_KEY","")
		if not key:
			p.error("no DREAM_API_KEY (env or ~/.dreamlabs/credentials) : set it or use --dry-run")
	done = {r["ep"] for r in mine(a,a.out)}
	todo = [e for e in range(a.episodes) if e not in done]
	print(f"{a.model} | {a.task} | tag : {a.tag} | referee : {a.referee} | {len(todo)} to run, {a.episodes - len(todo)} already in {a.out}",flush=True)
	if not todo:
		return 0
	out = Path(a.out)
	out.parent.mkdir(parents=True,exist_ok=True)
	run = Run(a,key)
	rc = 0
	try:
		for ep in todo:
			row = run.episode(ep)
			with out.open("a") as f:
				f.write(json.dumps(row) + "\n")
			lift = getattr(run.ref.verdict,"lift",None)
			print(
				f"ep {ep} : {'OK  ' if row['success'] else 'FAIL'} ({row['reason']})  sim {row['sim_s']:.1f} s  wall {row['wall_s']:.1f} s"
				f"  calls {row['calls']}  rtt {row['rtt_ms']} ms"
				+ (f"  lift {lift * 100:.1f} cm" if lift is not None else "")
				+ (f"\n  error : {row['error']}" if row["error"] else ""),
				flush=True,
			)
			if run.fatal:
				print("stopping : this error will not go away by itself, fix it and run the same command again to resume")
				rc = 1
				break
	except KeyboardInterrupt:
		print(f"\nstopped : finished episodes are in {a.out}, run the same command again to resume")
		rc = 130
	finally:
		run.close()
	got = mine(a,a.out)
	print(f"success : {sum(r['success'] for r in got)}/{len(got)}")
	return rc


if __name__ == "__main__":
	raise SystemExit(main())
