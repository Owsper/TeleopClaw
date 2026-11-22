"""A simulated hand in the tracker's place. It watches the scene like an operator and moves a virtual hand,
the real teleop code (clutch mapping, smoothing, rate limit, auto precision, pinch) turns that into arm motion.
Teleop-path demos with nobody at the webcam : not human demos.
	python -m dreamhands.bot data/dreamhands -n 40"""
import argparse,time

import numpy as np

from .recorder import Recorder
from .runner import TeleopRunner
from .save import Saver
from .scene import scatter
from .trials import APPROACH_Z,GRASP_Z,Referee
from .types import HandState

DT = .05
LO = np.array([.08,.08,.25])
HI = np.array([.92,.92,.9])
MID = np.array([.5,.5,.55])


class Hand:
	def __init__(self,r,rng):
		self.r,self.rng = r,rng

	def start(self):
		g = self.rng
		self.h = MID + g.uniform(-.1,.1,3) * [1,1,.5]
		self.pinch = 0.
		self.fist = 0
		self.wait = 4
		self.ph = 0  # over the apple, down, pinch, lift
		self.gain = g.uniform(.12,.22)
		self.vmax = g.uniform(.02,.04)
		self.zs = [APPROACH_Z + g.uniform(-.01,.01),GRASP_Z,GRASP_Z,g.uniform(.86,.875)]
		self.aim = g.normal(0,.0015,2)

	def goal(self):
		if self.ph < 2:
			self.xy = self.r.sim.data.body("apple").xpos[:2] + self.aim
		return np.r_[self.xy,self.zs[self.ph]]

	def read(self,now):
		g,t = self.rng,self.r.teleop
		if self.fist:
			# clutch : fist, hand back to the middle, hold still so the filter settles, let go
			self.fist -= 1
			if self.fist >= 8:
				self.h += (self.back - self.h) / (self.fist - 7)
		elif self.wait:
			self.wait -= 1
		elif self.ph == 2:
			self.pinch = min(1,self.pinch + .12)
			if self.pinch == 1:
				self.ph,self.wait = 3,int(g.integers(10,17))
		else:
			e = self.goal() - t.target
			if self.ph < 2 and abs(e[2]) < (.006,.004)[self.ph] and np.linalg.norm(e[:2]) < .004:
				self.ph += 1
				self.wait = int(g.integers(2,8))
			elif np.linalg.norm(e) > .0025:
				d = np.linalg.solve(t.cfg.axes,e) / t.cfg.scale / (t.cfg.precision if t.precise else 1) * self.gain
				h = self.h + d * min(1,self.vmax / np.linalg.norm(d))
				if np.any(h < LO) or np.any(h > HI):
					self.fist = 16
					self.back = self.h + (MID - self.h) * .8
				else:
					self.h = h
				if g.random() < .01:
					self.wait = int(g.integers(4,12))
		return HandState(t = now,present = True,confidence = .9,pos = self.h + g.normal(0,.0012,3),
			pinch = self.pinch,fist = self.fist > 0)


def episode(r,ref,hand,rng,rec = None,max_s = 20):
	def tick(keys = ""):
		now = r.step_i * DT
		return r.tick(hand.read(now),keys,now = now)
	r.sim.reset()
	scatter(r.sim,rng)
	r.engage = False
	r.go_ready()
	hand.start()
	r.tick(HandState(),now = r.step_i * DT)  # hand out of view : clears the filter
	for i in range(4):
		tick()
	r.engage = True
	r.record_images = rec is not None
	tick("r")
	ref.reset()
	for i in range(int(max_s / DT)):
		if ref(tick()).done:
			break
	r.flush()
	if rec:
		rec.stop()
	r.record_images = False
	return ref.verdict


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("root")
	ap.add_argument("-n",type = int,default = 40,help = "episodes to try")
	ap.add_argument("--task",default = "pick up the apple")
	ap.add_argument("--seed",type = int,default = 3000)
	a = ap.parse_args()
	r = TeleopRunner()
	ref = Referee(r.sim)
	sv = Saver(a.root)
	rec = Recorder(a.task,on_done = sv,referee = lambda: ref.verdict.success)
	r.on_tick.append(rec.on_tick)
	rng = np.random.default_rng(a.seed)
	hand = Hand(r,rng)
	t = time.perf_counter()
	for i in range(a.n):
		episode(r,ref,hand,rng,rec)
	r.close()
	r.sim.close()
	print(f"done : {sv.w.n} episodes in {a.root} ({time.perf_counter() - t:.0f} s)")


if __name__ == "__main__":
	main()
