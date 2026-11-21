import json
from types import SimpleNamespace as NS

import pytest
from dream.errors import DreamAuthError,DreamServerError
from shared.datasets import embodiments

from dreamhands import faas

MODEL = "dream-1-ft:ab12cd34-dreamhands"
DAY = "2026-11-21T"


def mkjob(status = "queued",**kw):
	return {"id": "ftj_1","dataset": "me/dreamhands-v1","suffix": "dreamhands","status": status,"queue_position": None,
		"hyperparameters": {"steps": 500},"progress": None,"fine_tuned_model": None,"error": None,
		"created_at": DAY + "22:00:00Z","started_at": None,"finished_at": None,**kw}

def prog(step,at):
	return {"phase": "training","step": step,"total_steps": 500,"loss": 0.5,"updated_at": f"{DAY}{at}Z"}

HAPPY = [mkjob(queue_position = 2),
	mkjob("running",started_at = DAY + "22:05:00Z",progress = prog(100,"22:10:00")),
	mkjob("running",started_at = DAY + "22:05:00Z",progress = prog(300,"22:20:00")),
	mkjob("succeeded"),
	mkjob("succeeded",fine_tuned_model = MODEL)]


class Fake:
	def __init__(self,ds = ("ready_to_process","processing","processed"),jobs = HAPPY,embs = (),ftErr = (),dsErr = None):
		self.calls = []
		self.ds = list(ds)
		self.jobs = list(jobs)
		self.embs = list(embs)
		self.ftErr = list(ftErr)
		self.dsErr = dsErr
		self.upErr = None

	def step(self,seq):
		return seq.pop(0) if len(seq) > 1 else seq[0]

	def dset(self,status,name = "dreamhands-v1"):
		return NS(id = "ds_1",name = f"me/{name}",embodiment = "me/yam-sim",status = status,size_bytes = 5e7,episodes = 30,
			error = self.dsErr if status == "failed" else None)

	def get_models(self):
		return [NS(id = "dreamlabs/dream-1-beta",owner = "public",servable = True,embodiment = "yam")]

	def validate_dataset(self,path,cfg):
		self.calls.append("validate")
		return NS(ok = True,error = None,sessions = 1,episodes = 30,frames = 3000,fps = 10,state_dim = 14,action_dim = 14,
			bytes = 5e7,video_keys = sorted(cfg["video"].values()),notes = [])

	def add_embodiment(self,name,cfg):
		self.calls.append("add_emb")
		if any(e.name == f"me/{name}" for e in self.embs): raise DreamServerError("exists",code = "conflict")
		self.embs.append(NS(name = f"me/{name}",embodiment_config = cfg))
		return self.embs[-1]

	def get_embodiments(self):
		self.calls.append("get_embs")
		return self.embs

	def upload_dataset(self,path,name,emb):
		self.calls.append("upload")
		if self.upErr: raise self.upErr
		return self.dset("ready_to_process",name)

	def get_dataset_status(self,id):
		self.calls.append("ds_status")
		return self.dset(self.step(self.ds))

	def get_datasets(self):
		return [self.dset("processed")]

	def retry_dataset(self,id):
		self.calls.append("retry")
		return self.dset("ready_to_process")

	def finetune(self,dataset,suffix,steps = None,**kw):
		self.calls.append("finetune")
		self.ftArgs = (dataset,suffix,steps,kw)
		if self.ftErr: raise self.ftErr.pop(0)
		return mkjob(queue_position = 2)

	def get_finetuning_status(self,id):
		self.calls.append("job_status")
		return self.step(self.jobs)

	def queue(self):
		return {"jobs_ahead_of_new": 3,"running": 2,"slots_free": 0,"slots_total": 2,"updated_at": DAY + "22:00:00Z"}

	def list(self):
		return [mkjob("running")]


def nonap(s):
	pass

@pytest.fixture
def path(tmp_path):
	return str(tmp_path / "faas.json")

@pytest.fixture
def st(path):
	return faas.St(path)


def test_happy_path(st,capsys):
	c,naps = Fake(),[]
	job = faas.run(c,st,"data/x",sleep = naps.append)
	assert job["fine_tuned_model"] == MODEL
	assert c.calls == ["validate","add_emb","upload"] + ["ds_status"] * 3 + ["finetune"] + ["job_status"] * 5
	assert c.ftArgs == ("me/dreamhands-v1","dreamhands",500,{})
	assert naps == [15,15,60,60,60,60]
	saved = json.load(open(st.path))
	assert (saved["embodiment"],saved["dataset_id"],saved["dataset_name"]) == ("me/yam-sim","ds_1","me/dreamhands-v1")
	assert (saved["job_id"],saved["fine_tuned_model"],saved["dataset_status"]) == ("ftj_1",MODEL,"processed")
	assert saved["t_upload"] and saved["t_job"] and saved["t_done"]
	out = capsys.readouterr().out
	assert "[DreamHands] fine-tune job ftj_1 on me/dreamhands-v1 | status : queued | queue pos : 2 | steps : 500 | ETA : unknown" in out
	assert "status : running | phase : training | step : 100/500 | loss : 0.5000 | ETA : ~20 min" in out
	assert "step : 300/500 | loss : 0.5000 | ETA : ~10 min" in out
	assert f'python -m dreamhands.evalrun --model {MODEL} --task "pick up the apple" --tag ft --episodes 10' in out
	assert "this key belongs to me, expected Nems1337" in out
	assert "dreamcli example yam-mujoco" in out and f"/model {MODEL}" in out and "pool_warming" in out

def test_emb_conflict_reuse(st):
	same = {**json.loads(json.dumps(faas.YAM_EMBODIMENT)),"head_camera": "overhead"}
	c = Fake(embs = [NS(name = "me/yam-sim",embodiment_config = same)])
	assert faas.embodiment(c,st,"yam-sim") == "me/yam-sim"
	assert c.calls == ["add_emb","get_embs"] and st["embodiment"] == "me/yam-sim"
	assert faas.embodiment(c,st,"yam-sim") == "me/yam-sim" and len(c.calls) == 2

def test_emb_conflict_other_config(st):
	other = {**faas.YAM_EMBODIMENT,"video": {"overhead": "observation.images.top"}}
	c = Fake(embs = [NS(name = "me/yam-sim",embodiment_config = other)])
	with pytest.raises(faas.Fail,match = "--emb yam-sim-2"): faas.embodiment(c,st,"yam-sim")
	assert "embodiment" not in st

def test_dataset_failed_then_retry(st):
	c = Fake(ds = ["processing","failed"],dsErr = "episode 3 : video too short")
	with pytest.raises(faas.Fail) as e: faas.run(c,st,"data/x",sleep = nonap)
	assert "video too short" in str(e.value) and "dreamhands.faas retry" in str(e.value)
	assert "finetune" not in c.calls and st["dataset_status"] == "failed" and not st.get("job_id")
	c.ds = ["processed"]
	faas.retry(c,st,nonap)
	assert "retry" in c.calls and st["dataset_status"] == "processed"
	faas.run(c,st,"data/x",sleep = nonap)
	assert c.calls.count("upload") == 1 and st["fine_tuned_model"] == MODEL

def test_payment_required(path,capsys):
	c = Fake(ftErr = [DreamServerError("the credits do not cover the job",code = "payment_required")])
	assert faas.main(["--state",path,"run","data/x"],c,nonap) == 1
	out = capsys.readouterr().out
	assert "no credits" in out and "$150" in out and "#support" in out
	st = faas.St(path)
	assert st["dataset_status"] == "processed" and not st.get("job_id")
	assert faas.main(["--state",path,"run","data/x"],c,nonap) == 0
	assert c.calls.count("upload") == 1 and c.calls.count("finetune") == 2 and faas.St(path)["fine_tuned_model"] == MODEL

def test_resume_from_state(path):
	faas.St(path).put(embodiment = "me/yam-sim",dataset_id = "ds_1",dataset_name = "me/dreamhands-v1",dataset_status = "processing")
	c = Fake(ds = ["processing","processed"])
	faas.run(c,faas.St(path),"data/x",wait = False,sleep = nonap)
	assert c.calls == ["ds_status","ds_status","finetune"]
	c = Fake()
	job = faas.run(c,faas.St(path),"data/x",sleep = nonap)
	assert set(c.calls) == {"job_status"} and job["fine_tuned_model"] == MODEL
	c = Fake(jobs = HAPPY[-1:])
	assert faas.main(["--state",path,"run"],c,nonap) == 0 and c.calls == ["job_status"]

def test_job_failed(st):
	c = Fake(jobs = [mkjob("running"),mkjob("failed",error = "the training did not finish")])
	with pytest.raises(faas.Fail,match = "the training did not finish"): faas.run(c,st,"data/x",sleep = nonap)
	assert st["job_status"] == "failed" and not st["fine_tuned_model"]
	c.jobs = list(HAPPY)
	faas.run(c,st,"data/x",sleep = nonap)
	assert c.calls.count("finetune") == 2 and c.calls.count("upload") == 1 and st["fine_tuned_model"] == MODEL

def test_retry_5xx_with_backoff(st):
	c,naps = Fake(ftErr = [DreamServerError("502",retryable = True)] * 2),[]
	st.put(dataset_name = "me/dreamhands-v1",dataset_status = "processed")
	faas.finetune(c,st,"dreamhands",500,naps.append,lr_schedule = "cosine",base_share = None)
	assert naps == [2,4] and c.calls.count("finetune") == 3 and st["job_id"] == "ftj_1"
	assert c.ftArgs[3] == {"lr_schedule": "cosine"}

def test_finetune_conflict(st):
	conflict = DreamServerError("suffix in use",code = "conflict")
	st.put(dataset_name = "me/dreamhands-v1",dataset_status = "processed")
	assert faas.finetune(Fake(ftErr = [conflict]),st,"dreamhands",500)["id"] == "ftj_1"
	with pytest.raises(faas.Fail,match = "--suffix other-2"): faas.finetune(Fake(ftErr = [conflict]),st,"other",500)

def test_ctrl_c_then_watch(path,capsys):
	def nap(s):
		if s == 60: raise KeyboardInterrupt
	assert faas.main(["--state",path,"run","data/x"],Fake(),nap) == 130
	assert faas.St(path)["job_id"] == "ftj_1" and "status --watch" in capsys.readouterr().out
	assert faas.main(["--state",path,"status","--watch"],Fake(),nonap) == 0
	assert faas.St(path)["fine_tuned_model"] == MODEL and "MODEL READY" in capsys.readouterr().out

def test_friendly_messages(path,tmp_path,capsys,monkeypatch):
	c = Fake()
	c.upErr = DreamServerError("name in use",code = "conflict")
	assert faas.main(["--state",path,"run","data/x"],c,nonap) == 1
	assert "--name dreamhands-v2" in capsys.readouterr().out
	c.upErr = DreamAuthError("no username",code = "forbidden")
	assert faas.main(["--state",path,"run","data/x"],c,nonap) == 1
	assert "no username yet" in capsys.readouterr().out
	monkeypatch.setenv("DREAM_CREDENTIALS",str(tmp_path / "none"))
	monkeypatch.delenv("DREAM_API_KEY",raising = False)
	assert faas.main(["--state",path,"models"]) == 1
	out = capsys.readouterr().out
	assert "DREAM_API_KEY=dl_live_" in out and "DREAM_DATA_URL=https://data.dreamlabs.ai" in out
	assert faas.main(["--state",path,"check",str(tmp_path / "nope")]) == 1
	assert "not a directory" in capsys.readouterr().out

def test_small_cmds(path,capsys):
	for cmd in ("models","queue","jobs"): assert faas.main(["--state",path,cmd],Fake()) == 0
	out = capsys.readouterr().out
	assert "dreamlabs/dream-1-beta | owner : public | servable : True | embodiment : yam" in out
	assert "3 ahead of a new job" in out and "slots free : 0/2" in out and "ftj_1 | running" in out
	assert faas.main(["--state",path,"status"],Fake()) == 1
	assert faas.main(["--state",path,"status","--job","ftj_1"],Fake(jobs = [mkjob("running")])) == 0
	out = capsys.readouterr().out
	assert out.strip().endswith("status : running | steps : 500 | ETA : unknown") and faas.St(path)["job_id"] == "ftj_1"
	with pytest.raises(SystemExit): faas.parse(["run","--steps","550"])

def test_eta():
	assert faas.eta(mkjob()) is None
	job = mkjob("running",started_at = DAY + "22:00:00Z",progress = prog(250,"22:10:00"))
	assert faas.eta(job) == 600
	assert faas.eta(job,(faas.ts(DAY + "22:05:00Z"),150)) == 750

def test_config_is_a_joints_only_body():
	body = embodiments.check_registration(faas.YAM_EMBODIMENT)
	embodiments.check_joints(body)
	assert (body.state_dim,body.action_dim,body.plan) == (14,14,None)
	with pytest.raises(embodiments.EmbodimentInvalidError): embodiments.check_eef(body)
