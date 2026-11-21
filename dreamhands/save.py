"""Recorder on_done glue : judge the episode, log the verdict, write it only when KEEP."""
from . import quality
from .lerobot import Writer


class Saver:
	def __init__(self,root,cfg=None):
		self.w = Writer(root)
		self.cfg = cfg
		self.qlog = str(root).rstrip("/\\") + ".quality.jsonl"  # beside the dataset, never inside it
		self.last = None

	def __call__(self,ep):
		v = quality.judge(ep,self.cfg)
		idx = self.w.add(ep) if v.keep else None
		quality.log(self.qlog,v,idx)
		self.last = v
		print(f"episode {'-' if idx is None else idx} : {v}  ({ep.dur:.1f} s, kept {self.w.n})")
		return v
