"""LeRobot v2.1 writer by hand (pyarrow + av), layout held to Dream's local dataset check."""
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARTS = {"right_arm": [0,6],"right_gripper": [6,7],"left_arm": [7,13],"left_gripper": [13,14]}
YAM_EMBODIMENT = {"state": PARTS,"action": PARTS,"video": {r: f"observation.images.{r}" for r in ("overhead","wrist_l","wrist_r")}}
VKEY = {"top": "observation.images.overhead","left": "observation.images.wrist_l","right": "observation.images.wrist_r"}
DIM = 14
CHUNK = 1000
DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
NAMES = [f"{s}_{j}" for s in ("right","left") for j in ("j1","j2","j3","j4","j5","j6","gripper")]


def rdl(p):
	return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def apl(p,row):
	with open(p,"a",newline = "\n") as f:
		f.write(json.dumps(row) + "\n")


# dot files are skipped by the validator and the upload, so a half written one never counts
def tmp(p):
	return p.with_name("." + p.name)


def stat(a,shape,cnt):
	a = np.asarray(a,np.float64).reshape(-1,shape[0])
	st = {k: getattr(a,k)(0).reshape(shape).tolist() for k in ("min","max","mean","std")}
	st["count"] = [cnt]
	return st


def encVid(path,imgs,fps):
	h,w = imgs[0].shape[:2]
	with av.open(str(path),"w") as out:
		# bf 0 : no reorder, so no edit list and pts start at 0
		st = out.add_stream("libx264",rate = fps,options = {"g": "2","bf": "0","crf": "23"})
		st.width,st.height,st.pix_fmt = w,h,"yuv420p"
		for img in imgs:
			out.mux(st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img),format = "rgb24")))
		out.mux(st.encode())


def mkInfo(ep,fps):
	vec = {"dtype": "float32","shape": [DIM],"names": NAMES}
	ft = {"action": vec,"observation.state": vec}
	for cam,key in VKEY.items():
		h,w = ep.images[cam][0].shape[:2]
		ft[key] = {"dtype": "video","shape": [h,w,3],"names": ["height","width","channels"],"info": {
			"video.height": h,"video.width": w,"video.codec": "h264","video.pix_fmt": "yuv420p",
			"video.is_depth_map": False,"video.fps": fps,"video.channels": 3,"has_audio": False}}
	ft["timestamp"] = {"dtype": "float32","shape": [1],"names": None}
	for k in ("frame_index","episode_index","index","task_index"):
		ft[k] = {"dtype": "int64","shape": [1],"names": None}
	return {"codebase_version": "v2.1","robot_type": "yam","total_episodes": 0,"total_frames": 0,"total_tasks": 0,
		"total_videos": 0,"total_chunks": 0,"chunks_size": CHUNK,"fps": fps,"splits": {},
		"data_path": DATA,"video_path": VIDEO,"features": ft}


class Writer:
	def __init__(self,root,fps = 10):
		self.root = Path(root)
		self.meta = self.root / "meta"
		self.meta.mkdir(parents = True,exist_ok = True)
		self.fps = fps
		self.info = None
		eps = rdl(self.meta / "episodes.jsonl")
		self.n = len(eps)
		self.frames = sum(e["length"] for e in eps)
		self.tasks = [t["task"] for t in rdl(self.meta / "tasks.jsonl")]
		if (self.meta / "info.json").exists():
			self.info = json.loads((self.meta / "info.json").read_text())
			self.fps = self.info["fps"]
			self.save()

	def save(self):
		self.info.update(total_episodes = self.n,total_frames = self.frames,total_tasks = len(self.tasks),
			total_videos = self.n * len(VKEY),total_chunks = -(-self.n // CHUNK),splits = {"train": f"0:{self.n}"})
		p = self.meta / "info.json"
		tmp(p).write_text(json.dumps(self.info,indent = 4))
		os.replace(tmp(p),p)

	def add(self,ep):
		assert ep.fps == self.fps,f"episode fps : {ep.fps}, dataset fps : {self.fps}"
		i,n = self.n,ep.n
		if not self.info:
			self.info = mkInfo(ep,self.fps)
		ti = self.tasks.index(ep.task) if ep.task in self.tasks else len(self.tasks)
		fi = np.arange(n,dtype = np.int64)
		cols = {
			"action": np.asarray(ep.action,np.float32),
			"observation.state": np.asarray(ep.state,np.float32),
			"timestamp": (fi / self.fps).astype(np.float32),
			"frame_index": fi,
			"episode_index": np.full(n,i,np.int64),
			"index": self.frames + fi,
			"task_index": np.full(n,ti,np.int64),
		}
		tbl = pa.table({k: pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()),DIM) if v.ndim == 2 else pa.array(v) for k,v in cols.items()})
		st = {k: stat(v,v.shape[1:] or (1,),n) for k,v in cols.items()}
		for cam,key in VKEY.items():
			px = np.stack(ep.images[cam][::max(1,n // 20)])[:,::4,::4]
			st[key] = stat(px / 255,(3,1,1),len(px))

		ck = i // CHUNK
		pp = self.root / DATA.format(episode_chunk = ck,episode_index = i)
		vps = {cam: self.root / VIDEO.format(episode_chunk = ck,video_key = key,episode_index = i) for cam,key in VKEY.items()}
		for p in (pp,*vps.values()):
			p.parent.mkdir(parents = True,exist_ok = True)
		pq.write_table(tbl,tmp(pp))
		for cam,p in vps.items():
			encVid(tmp(p),ep.images[cam],self.fps)
		for p in (pp,*vps.values()):
			os.replace(tmp(p),p)

		# episodes.jsonl line is the commit, info.json is rebuilt from it on resume
		apl(self.meta / "episodes_stats.jsonl",{"episode_index": i,"stats": st})
		if ti == len(self.tasks):
			self.tasks.append(ep.task)
			apl(self.meta / "tasks.jsonl",{"task_index": ti,"task": ep.task})
		apl(self.meta / "episodes.jsonl",{"episode_index": i,"tasks": [ep.task],"length": n})
		self.n += 1
		self.frames += n
		self.save()
		return i


def merge(src,dst):
	"""append every episode of the dataset src (another laptop's recordings) to dst"""
	src = Path(src)
	si = json.loads((src / "meta" / "info.json").read_text())
	w = Writer(dst)
	if w.info is None:
		w.info = {k: v for k,v in si.items()}
	assert si["fps"] == w.fps,f"fps : {si['fps']} vs {w.fps}"
	assert si["features"] == w.info["features"],"features differ (camera size or keys)"
	tasks = {t["task_index"]: t["task"] for t in rdl(src / "meta" / "tasks.jsonl")}
	stats = {s["episode_index"]: s["stats"] for s in rdl(src / "meta" / "episodes_stats.jsonl")}
	vkeys = [k for k,f in si["features"].items() if f["dtype"] == "video"]
	for e in rdl(src / "meta" / "episodes.jsonl"):
		j,i,n = e["episode_index"],w.n,e["length"]
		sck,ck = j // si["chunks_size"],i // CHUNK
		tbl = pq.read_table(src / si["data_path"].format(episode_chunk = sck,episode_index = j))
		ti,new = {},[]
		for t in sorted(set(tbl["task_index"].to_pylist())):
			s = tasks[t]
			if s not in w.tasks and s not in new:
				new.append(s)
			ti[t] = w.tasks.index(s) if s in w.tasks else len(w.tasks) + new.index(s)
		fi = np.asarray(tbl["frame_index"].to_pylist(),np.int64)
		cols = {"episode_index": np.full(n,i,np.int64),"index": w.frames + fi,
			"task_index": np.asarray([ti[t] for t in tbl["task_index"].to_pylist()],np.int64)}
		for k,v in cols.items():
			tbl = tbl.set_column(tbl.schema.get_field_index(k),k,pa.array(v))
		st = dict(stats[j])
		st.update({k: stat(v,(1,),n) for k,v in cols.items()})
		pp = w.root / DATA.format(episode_chunk = ck,episode_index = i)
		pp.parent.mkdir(parents = True,exist_ok = True)
		pq.write_table(tbl,tmp(pp))
		vps = []
		for k in vkeys:
			vp = w.root / VIDEO.format(episode_chunk = ck,video_key = k,episode_index = i)
			vp.parent.mkdir(parents = True,exist_ok = True)
			tmp(vp).write_bytes((src / si["video_path"].format(episode_chunk = sck,video_key = k,episode_index = j)).read_bytes())
			vps.append(vp)
		for p in (pp,*vps):
			os.replace(tmp(p),p)
		apl(w.meta / "episodes_stats.jsonl",{"episode_index": i,"stats": st})
		for s in new:
			apl(w.meta / "tasks.jsonl",{"task_index": len(w.tasks),"task": s})
			w.tasks.append(s)
		apl(w.meta / "episodes.jsonl",{"episode_index": i,"tasks": e["tasks"],"length": n})
		w.n += 1
		w.frames += n
		w.save()
	return w.n


def validate(root):
	from dream import datasets
	return datasets.validate_dataset(root,YAM_EMBODIMENT)


if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser(description = "check a dataset, --add appends other datasets to it first")
	ap.add_argument("root")
	ap.add_argument("--add",nargs = "+",default = [],metavar = "SRC")
	a = ap.parse_args()
	for s in a.add:
		print(f"merged {s} : {merge(s,a.root)} episodes now")
	rep = validate(a.root)
	for k in ("ok","error","episodes","frames","fps","video_keys","notes"):
		print(f"{k} : {getattr(rep,k)}")
	sys.exit(0 if rep.ok else 1)
