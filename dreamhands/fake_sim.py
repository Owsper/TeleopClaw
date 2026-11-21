"""FakeSim : Tick source with no MuJoCo / webcam / GL, same shape as the runner's ticks."""
import numpy as np

from .episode import CAMS,HZ
from .types import HandState,Tick

H,W = 360,640
JUMP = 1.0  # rad, on joint 1


class FakeSim:
	def __init__(self,seconds=5,seed=0,lost=0,ik=0,jump_at=None,events=None):
		self.n = int(round(seconds * HZ))
		self.jump_at = jump_at
		self.events = events or {}
		rng = np.random.default_rng(seed)
		self.amp = rng.uniform(0.1,0.4,14)
		self.ph = rng.uniform(0,6.28,14)
		self.lost = self.pick(rng,lost)
		self.ik = self.pick(rng,ik)
		yy,xx = np.mgrid[0:H,0:W]
		self.bg = np.stack([xx * 255 // W,yy * 255 // H,(xx + yy) * 255 // (W + H)],-1).astype(np.uint8)

	def pick(self,rng,frac):
		return set(rng.choice(self.n,int(round(frac * self.n)),replace=False).tolist())

	def q(self,step):
		t = step / HZ
		v = self.amp * np.sin(0.8 * t + self.ph)
		v[[6,13]] = 0.5 + 0.5 * np.sin(0.8 * t + self.ph[[6,13]])
		if self.jump_at is not None and step >= self.jump_at:
			v[1] += JUMP
		return v

	def img(self,step,k):
		im = np.roll(self.bg,k,2)
		x = (step * 6 + k * 150) % (W - 80)
		y = 100 + k * 60
		im[y:y + 60,x:x + 80] = 255
		return im

	def ticks(self):
		for i in range(self.n):
			t = i / HZ
			imgs = None
			if i % 2 == 0:
				imgs = {c: self.img(i,k) for k,c in enumerate(CAMS)}
			lost = i in self.lost
			flags = {"tracking_lost": lost,"ik_reject": i in self.ik,"workspace_edge": False}
			hand = HandState(t=t,present=not lost,confidence=0.0 if lost else 1.0)
			# action = next step's pose, so it leads the state a bit
			yield Tick(step=i,t=t,state=self.q(i),action=self.q(i + 1),images=imgs,hand=hand,
				mode="ENGAGED",flags=flags,events=list(self.events.get(i,[])))

	def run(self,fn):
		for tick in self.ticks():
			fn(tick)
