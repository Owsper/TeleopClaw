"""Episode : what the recorder buffers, quality judges and the lerobot writer saves.

Frames are the ticks that carry images (even steps -> 10 fps), ticks are every 20 Hz step.
Vectors are 14-D right-first (see types.py). Images are RGB uint8 HxWx3 per camera.
"""
from dataclasses import dataclass,field

import numpy as np

CAMS = ("top","left","right")
FPS = 10
HZ = 20


@dataclass
class Episode:
	task: str = ""
	fps: int = FPS
	hz: int = HZ
	# per frame
	state: list = field(default_factory=list)
	action: list = field(default_factory=list)
	images: dict = field(default_factory=lambda: {c: [] for c in CAMS})
	# per tick, for quality
	tstate: list = field(default_factory=list)
	taction: list = field(default_factory=list)
	flags: list = field(default_factory=list)
	modes: list = field(default_factory=list)
	success: bool | None = None  # referee verdict, None = nobody judged
	pend: bool = False

	def add(self,tick):
		st = np.asarray(tick.state,np.float32)
		act = np.asarray(tick.action,np.float32)
		self.tstate.append(st)
		self.taction.append(act)
		self.flags.append(dict(tick.flags))
		self.modes.append(tick.mode)
		# a frame's action is the command sent right after its observation = next tick's
		if self.pend:
			self.action[-1] = act
			self.pend = False
		if tick.images is not None:
			self.state.append(st)
			self.action.append(act)
			for c in CAMS:
				self.images[c].append(tick.images[c])
			self.pend = True

	@property
	def n(self):
		return len(self.state)

	@property
	def dur(self):
		return self.n / self.fps
