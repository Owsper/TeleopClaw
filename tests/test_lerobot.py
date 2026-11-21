import json
import time

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from dreamhands import lerobot
from dreamhands.episode import Episode
from dreamhands.lerobot import VKEY,YAM_EMBODIMENT,Writer,validate


def mkEp(n = 50,task = "pick up the cube"):
	rng = np.random.default_rng(n)
	ep = Episode(task = task)
	for i in range(n):
		ep.state.append(rng.random(14,dtype = np.float32))
		ep.action.append(rng.random(14,dtype = np.float32))
		for k,c in enumerate(ep.images):
			img = np.full((360,640,3),i * 5,np.uint8)
			img[:20,:20] = (255,0,0) if k == 0 else (0,255,0)
			ep.images[c].append(img)
	return ep


def test_5s(tmp_path):
	root = tmp_path / "sess"
	w = Writer(root)
	t = time.perf_counter()
	assert w.add(mkEp()) == 0
	print(f"write : {time.perf_counter() - t:.2f} s")
	rep = validate(root)
	print(rep)
	assert rep.ok,rep.error
	assert (rep.episodes,rep.frames,rep.fps,rep.state_dim,rep.action_dim) == (1,50,10,14,14)
	assert rep.video_keys == sorted(VKEY.values())
	assert rep.notes == []

	for key in VKEY.values():
		with av.open(str(root / f"videos/chunk-000/{key}/episode_000000.mp4")) as c:
			vs = c.streams.video[0]
			frs = list(c.decode(vs))
			assert (vs.codec_context.name,vs.pix_fmt,vs.average_rate) == ("h264","yuv420p",10)
		assert [f.time for f in frs] == pytest.approx(np.arange(50) / 10)
		# gray level is the frame number, so order survived the encode
		assert [f.to_ndarray(format = "rgb24")[200,300,0] for f in frs] == pytest.approx(np.arange(50) * 5,abs = 4)

	tbl = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
	assert tbl.column_names == ["action","observation.state","timestamp","frame_index","episode_index","index","task_index"]
	ty = tbl.schema.field("action").type
	assert (ty.list_size,str(ty.value_type),str(tbl.schema.field("timestamp").type)) == (14,"float","float")
	st = json.loads((root / "meta/episodes_stats.jsonl").read_text())["stats"]
	assert np.shape(st["action"]["mean"]) == (14,)
	assert np.shape(st[VKEY["top"]]["mean"]) == (3,1,1)
	assert 0 <= st[VKEY["top"]]["max"][0][0][0] <= 1


def test_append_resume(tmp_path):
	root = tmp_path / "sess"
	w = Writer(root)
	w.add(mkEp())
	assert w.add(mkEp(30,"put the cube in the bin")) == 1
	rep = validate(root)
	assert rep.ok,rep.error
	assert (rep.episodes,rep.frames) == (2,80)

	w = Writer(root)
	assert w.n == 2
	assert w.add(mkEp()) == 2
	rep = validate(root)
	assert rep.ok,rep.error
	assert (rep.episodes,rep.frames,rep.notes) == (3,130,[])
	info = json.loads((root / "meta/info.json").read_text())
	assert (info["total_tasks"],info["total_videos"],info["splits"]) == (2,9,{"train": "0:3"})
	assert pq.read_table(root / "data/chunk-000/episode_000002.parquet")["index"].to_pylist() == list(range(80,130))


def test_crash(tmp_path,monkeypatch):
	root = tmp_path / "sess"
	w = Writer(root)
	w.add(mkEp())

	def boom(path,imgs,fps):
		path.write_bytes(b"half a video")
		raise RuntimeError("crash")
	monkeypatch.setattr(lerobot,"encVid",boom)
	with pytest.raises(RuntimeError):
		w.add(mkEp())
	monkeypatch.undo()
	rep = validate(root)
	assert rep.ok,rep.error
	assert rep.episodes == 1
	assert Writer(root).add(mkEp()) == 1
	assert validate(root).episodes == 2


# 1.6 s rule : at 10 fps an episode needs 48 * 10 // 30 + 1 = 17 frames
def test_short(tmp_path):
	root = tmp_path / "sess"
	Writer(root).add(mkEp(10))
	rep = validate(root)
	print(rep.error)
	assert not rep.ok
	assert "shorter than one training sample" in rep.error


def test_embodiment(monkeypatch):
	try:
		import dream_examples.yam_mujoco as ex
		# dream_control imports its neighbour yam_env as a top level module
		monkeypatch.syspath_prepend(list(ex.__path__)[0])
		from dream_examples.yam_mujoco.dream_control import YAM_EMBODIMENT as ref
	except Exception:
		pytest.skip("dream_examples not importable")
	assert YAM_EMBODIMENT == ref
