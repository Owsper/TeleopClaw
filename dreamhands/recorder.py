"""Recorder : buffers ticks into an Episode, driven by tick.events or start() / stop()."""
from .episode import Episode


class Recorder:
	def __init__(self,task,on_done=None,referee=None,max_s=60):
		self.task = task
		self.on_done = on_done
		self.referee = referee
		self.max_s = max_s
		self.ep = None
		self.done = 0
		self.dropped = 0

	def start(self):
		if self.ep is not None:
			self.dropped += 1
		self.ep = Episode(self.task)

	def stop(self,keep=True):
		ep,self.ep = self.ep,None
		if ep is None:
			return None
		if not keep:
			self.dropped += 1
			return None
		if self.referee:
			ep.success = bool(self.referee())
		self.done += 1
		if self.on_done:
			self.on_done(ep)
		return ep

	def on_tick(self,tick):
		ev = tick.events
		if "record" in ev:
			self.start()
		if self.ep is None:
			return
		if "discard" in ev:
			self.stop(False)
			return
		self.ep.add(tick)
		# raw frames eat RAM, cut long episodes
		if "next" in ev or len(self.ep.tstate) >= self.max_s * self.ep.hz:
			self.stop()


def demo():
	import numpy as np

	from .episode import CAMS
	from .fake_sim import FakeSim

	eps = []
	rec = Recorder("pick the cube",on_done=eps.append)
	FakeSim(seconds=5,events={0: ["record"],99: ["next"]}).run(rec.on_tick)
	ep = eps[0]
	st,act = np.array(ep.state),np.array(ep.action)
	nt = len(ep.tstate)
	fps = ep.n / (nt / ep.hz)
	print(f"frames : {ep.n}")
	print(f"fps : {fps:g}")
	print(f"duration : {ep.dur:g} s")
	print(f"state : {st.shape}  action : {act.shape}")
	for c in CAMS:
		print(f"{c} : {len(ep.images[c])} frames {ep.images[c][0].shape} {ep.images[c][0].dtype}")
	print(f"ticks : {nt}")
	ok = ep.n == 50 and nt == 100 and fps == ep.fps == 10 and st.shape == act.shape == (50,14)
	ok = ok and len(ep.images) == 3 and all(len(v) == 50 for v in ep.images.values())
	print("OK" if ok else "FAIL")
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(demo())
