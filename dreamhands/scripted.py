"""Scripted apple picks at random spots, recorded like teleop : recorder -> quality gate -> LeRobot.
Aim, hover height and speed vary per episode. Seeds start at 1000 so the eval spots (0..N) stay unseen.
	python -m dreamhands.scripted data/dreamhands -n 50"""
import argparse,time

import numpy as np

from .ik import TOP_DOWN
from .recorder import Recorder
from .runner import TeleopRunner
from .save import Saver
from .scene import scatter
from .trials import APPROACH_Z,GRASP_Z,Referee
from .types import HandState,Tick,to14

PLAN = [(None,1,40),(GRASP_Z,1,30),(GRASP_Z,0,15),(None,0,30),(None,0,40)]


class Demo:
	def __init__(self,root,task = "pick up the apple"):
		self.r = TeleopRunner(async_images = False)
		self.ref = Referee(self.r.sim)
		self.sv = Saver(root)
		self.rec = Recorder(task,on_done = self.sv,referee = lambda: self.ref.verdict.success)
		self.i = 0

	def emit(self,ev = ()):
		r = self.r
		st = r.sim.step(r.left,r.right_cmd())
		im = r.sim.images() if self.rec.ep is not None and self.i % 2 == 0 else None
		t = Tick(step = self.i,t = self.i * .05,state = to14(st["right"],st["left"]),action = to14(r.right_cmd(),r.left),
			images = im,hand = HandState(),mode = "ENGAGED",tcp = r.tcp(),
			flags = {"ik_reject": False,"tracking_lost": False},events = list(ev))
		self.rec.on_tick(t)
		self.i += 1
		return t

	def ep(self,rng):
		r = self.r
		r.sim.reset()
		scatter(r.sim,rng)
		r.go_ready()
		xy = r.sim.data.body("apple").xpos[:2] + rng.normal(0,.002,2)
		hz = APPROACH_Z + rng.uniform(-.01,.01)
		k = rng.uniform(.8,1.3)
		cur = r.tcp()
		self.emit(["record"])
		self.ref.reset()
		for z,g,n in PLAN:
			p = np.r_[xy,hz if z is None else z]
			n = int(n * k)
			for j in range(1,n + 1):
				res,rej = r.ik.step(cur + (p - cur) * min(1,j / max(1,n - 5)),TOP_DOWN,r.q)
				if not rej:
					r.q = res.q
				r.grip = float(g)
				if self.rec.ep is not None and self.ref(self.emit()).done:
					self.rec.stop()
					return self.ref.verdict
			cur = p
		if self.rec.ep is not None:
			self.emit(["next"])
		return self.ref.verdict


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("root")
	ap.add_argument("-n",type = int,default = 50,help = "episodes to try")
	ap.add_argument("--task",default = "pick up the apple")
	ap.add_argument("--seed",type = int,default = 1000)
	a = ap.parse_args()
	d = Demo(a.root,a.task)
	rng = np.random.default_rng(a.seed)
	t = time.perf_counter()
	for i in range(a.n):
		d.ep(rng)
	d.r.sim.close()
	print(f"done : {d.sv.w.n} episodes in {a.root} ({time.perf_counter() - t:.0f} s)")


if __name__ == "__main__":
	main()
