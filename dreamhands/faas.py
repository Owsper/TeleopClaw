"""Dream Labs FaaS chain : LeRobot session -> embodiment -> upload -> fine-tune -> model id.
State lives in data/faas.json, every step can be re-run. python -m dreamhands.faas -h"""
import argparse,json,os,re,sys,time
from datetime import datetime,timezone

from dream import fine_tuning
from dream.errors import (DreamAuthError,DreamConfigError,DreamDatasetError,DreamError,
	DreamProtocolError,DreamServerError,DreamUploadError)

P = {"right_arm": [0,6],"right_gripper": [6,7],"left_arm": [7,13],"left_gripper": [13,14]}
YAM_EMBODIMENT = {"state": P,"action": P,"video": {r: f"observation.images.{r}" for r in ("overhead","wrist_l","wrist_r")}}

STATE = "data/faas.json"
DIR = "data/dreamhands"
TASK = "pick up the apple"
USER = "veyselpavyoncuoglu"  # Dream Labs account, embodiments + datasets are <USER>/<name>
ME = "python -m dreamhands.faas"
DEAD = ("failed","cancelled")
DS_POLL = 15
JOB_POLL = 60

NO_KEY = f"""no API key. write {{path}} (only the key line is required, the URLs are the defaults) :
	DREAM_API_KEY=dl_live_...
	DREAM_GATEWAY_URL=https://api.dreamlabs.ai
	DREAM_API_URL=https://platform.dreamlabs.ai
	DREAM_DATA_URL=https://data.dreamlabs.ai
or set DREAM_API_KEY in the environment. then check it : {ME} models"""
NO_CREDITS = "no credits : post the $150 claim in #support, ask the mentors. then run the same command again (the dataset is kept)"


class Fail(Exception):
	pass


class St(dict):
	def __init__(self,path = STATE):
		self.path = path
		if os.path.exists(path):
			with open(path) as f: self.update(json.load(f))

	def put(self,**kw):
		self.update(kw)
		os.makedirs(os.path.dirname(self.path) or ".",exist_ok = True)
		with open(self.path + ".tmp","w") as f: json.dump(self,f,indent = 1)
		os.replace(self.path + ".tmp",self.path)


def now():
	return datetime.now(timezone.utc).isoformat(timespec = "seconds")

def ts(s):
	return datetime.fromisoformat(s.replace("Z","+00:00")).timestamp() if s else None

def tail(s):
	return (s or "").split("/")[-1]

def full(s):
	return s if not s or "/" in s else f"{USER}/{s}"

def bump(s):
	m = re.search(r"\d+$",s)
	return s[:m.start()] + str(int(m.group()) + 1) if m else s + "-2"

def flaky(e):
	return isinstance(e,(DreamProtocolError,DreamUploadError)) or getattr(e,"retryable",False)

def call(fn,*a,sleep = time.sleep,tries = 5,**kw):
	for i in range(tries):
		try:
			return fn(*a,**kw)
		except DreamError as e:
			if not flaky(e) or i == tries - 1: raise
			wait = getattr(e,"retry_after_s",None) or 2 ** min(i + 1,6)
			print(f"retry {i + 1} in {wait:.0f} s : {e}")
			sleep(wait)

def explain(e):
	code = getattr(e,"code",None)
	if isinstance(e,DreamAuthError):
		if code == "forbidden": return "forbidden : the account has no username yet, ask Dream Labs support (the mentors) to set one"
		return f"key refused : check DREAM_API_KEY in ~/.dreamlabs/credentials, a key only works on the site that made it\n{e}"
	if code == "payment_required": return NO_CREDITS
	if isinstance(e,DreamConfigError) and "no api key" in str(e): return noKey()
	if isinstance(e,DreamDatasetError): return f"dataset check failed : {e}"
	if isinstance(e,DreamUploadError): return f"upload did not finish : {e}\nrun the same command again, it reuses the dataset"
	if isinstance(e,DreamProtocolError): return f"cannot reach Dream Labs : {e}\ncheck the network, run the same command again (state is saved)"
	if getattr(e,"retryable",False): return f"Dream Labs server error, still failing after retries : {e}\nwait a minute, run the same command again"
	return f"{e} ({code})" if code else str(e)

def noKey():
	from dream import credentials
	return NO_KEY.format(path = credentials.file_path())

def client(needKey = True):
	import dream
	from dream import credentials
	if needKey and not credentials.load().api_key: raise Fail(noKey())
	return dream.DreamClient()


# ---- steps

def models(c,sleep = time.sleep):
	for m in call(c.get_models,sleep = sleep):
		print(f"{m.id} | owner : {m.owner} | servable : {m.servable} | embodiment : {m.embodiment}")
	print("key ok")

def check(c,path):
	r = c.validate_dataset(path,YAM_EMBODIMENT)
	if not r.ok: raise Fail(f"dataset check failed : {r.error}")
	print(f"check ok : {r.sessions} session(s), {r.episodes} episodes, {r.frames} frames, {r.fps} fps, "
		f"state/action {r.state_dim}/{r.action_dim}, {r.bytes / 1e6:.1f} MB, videos : {', '.join(r.video_keys)}")
	for n in r.notes: print("note :",n)
	return r

def sameCfg(a,b):
	norm = lambda cfg: json.loads(json.dumps({"head_camera": "overhead",**cfg}))
	return norm(a) == norm(b)

def embodiment(c,st,name,sleep = time.sleep):
	if tail(st.get("embodiment")) == name:
		print("embodiment :",st["embodiment"],"(state file)")
		return st["embodiment"]
	try:
		emb = call(c.add_embodiment,name,YAM_EMBODIMENT,sleep = sleep)
		print("embodiment created :",emb.name)
	except DreamServerError as e:
		if e.code != "conflict": raise
		emb = next((x for x in call(c.get_embodiments,sleep = sleep) if tail(x.name) == name),None)
		if not emb: raise
		if not sameCfg(emb.embodiment_config,YAM_EMBODIMENT):
			raise Fail(f"embodiment {emb.name} exists with another config : use --emb {bump(name)}\n"
				f"theirs : {json.dumps(emb.embodiment_config)}\nours : {json.dumps(YAM_EMBODIMENT)}") from None
		print("embodiment exists with the same config, reusing :",emb.name)
	if emb.name.rpartition("/")[0] != USER:
		print(f"warning : this key belongs to {emb.name.rpartition('/')[0]}, expected {USER} (wrong key ?)")
	st.put(embodiment = emb.name,t_emb = now())
	return emb.name

def upload(c,st,path,name,sleep = time.sleep):
	print(f"uploading {path} as {name} under {st['embodiment']} (zip + S3, can take a while) ...")
	try:
		ds = call(c.upload_dataset,path,name,st["embodiment"],sleep = sleep)
	except DreamServerError as e:
		if e.code != "conflict": raise
		old = next((d for d in call(c.get_datasets,sleep = sleep) if tail(d.name) == name),None)
		there = f"\nto fine-tune on the one already there ({old.id}, {old.status}) : {ME} finetune --dataset {old.name}" if old else ""
		raise Fail(f"dataset name {name} is taken (already uploaded / processed) : use --name {bump(name)}{there}") from None
	st.put(dataset_id = ds.id,dataset_name = ds.name,dataset_status = ds.status,t_upload = now(),t_processed = None,
		job_id = None,job_status = None,fine_tuned_model = None,t_job = None,t_done = None)
	print(f"uploaded : {ds.name} ({ds.id}), {ds.episodes} episodes, {ds.size_bytes / 1e6:.1f} MB, status : {ds.status}")
	return ds

def waitDataset(c,st,sleep = time.sleep,every = DS_POLL):
	if not st.get("dataset_id"): raise Fail(f"no dataset in the state file : {ME} run {DIR}")
	last = None
	while True:
		ds = call(c.get_dataset_status,st["dataset_id"],sleep = sleep,tries = 999)
		if ds.status != last:
			last = ds.status
			print(f"dataset {ds.name} : {ds.status}")
			st.put(dataset_status = ds.status,dataset_name = ds.name)
		if ds.status == "processed":
			st.put(t_processed = now())
			return ds
		if ds.status in ("failed","deleted"):
			raise Fail(f"dataset {ds.name} {ds.status} : {ds.error}\n"
				f"retry the processing (nothing is sent again) : {ME} retry\n"
				f"or fix the data and upload again : {ME} run <dir> --name {bump(tail(ds.name))}")
		sleep(every)

def eta(job,ref = None):
	p = job.get("progress") or {}
	step,total = p.get("step"),p.get("total_steps")
	if not step or not total: return None
	t = ts(p.get("updated_at")) or time.time()
	t0,s0 = ref if ref and step > ref[1] else (ts(job.get("started_at")),0)
	if not t0 or t <= t0: return None
	return (total - step) * (t - t0) / (step - s0)

def chat(job,ref = None):
	p = job.get("progress") or {}
	total = p.get("total_steps") or (job.get("hyperparameters") or {}).get("steps") or "?"
	out = [f"[DreamHands] fine-tune job {job['id']} on {job.get('dataset')}",f"status : {job['status']}"]
	if job.get("queue_position") is not None: out.append(f"queue pos : {job['queue_position']}")
	if p.get("phase"): out.append(f"phase : {p['phase']}")
	out.append(f"step : {p['step']}/{total}" if p.get("step") is not None else f"steps : {total}")
	if p.get("loss") is not None: out.append(f"loss : {p['loss']:.4f}")
	if job.get("fine_tuned_model"): return " | ".join(out + [f"model : {job['fine_tuned_model']}"])
	if job["status"] == "succeeded": return " | ".join(out + ["model : being published"])
	left = eta(job,ref)
	return " | ".join(out + ["ETA : unknown" if left is None else f"ETA : ~{left / 60:.0f} min"])

def ready(m):
	print(f"""
MODEL READY : {m}
eval :
	python -m dreamhands.evalrun --model {m} --task "{TASK}" --tag ft --episodes 10 --out data/eval.jsonl --scatter
stock sim :
	dreamcli example yam-mujoco
	/model {m}
note : a fresh fine-tune answers pool_warming for 25-50 min on first use, start it early and keep retrying""")

def finetune(c,st,suffix,steps,sleep = time.sleep,dataset = None,**knobs):
	ds = dataset or st.get("dataset_name")
	if not ds: raise Fail(f"no dataset in the state file : {ME} run {DIR}, or pass --dataset {USER}/<name>")
	knobs = {k: v for k,v in knobs.items() if v is not None}
	try:
		job = call(c.finetune,ds,suffix,steps = steps,sleep = sleep,**knobs)
	except DreamServerError as e:
		if e.code == "payment_required" or "credit" in str(e).lower(): raise Fail(NO_CREDITS) from None
		if e.code != "conflict": raise
		# a job the service already has for us (lost answer, other machine) : take it
		job = next((j for j in getattr(c,"list",fine_tuning.list)() if j.get("suffix") == suffix
			and j.get("dataset") == ds and j.get("status") not in DEAD),None)
		if not job:
			raise Fail(f"fine-tune refused (conflict) : {e}\nsuffix {suffix} in use, dataset not processed yet, or too many active jobs."
				f" try --suffix {bump(suffix)}, see {ME} jobs") from None
		print(f"job {job['id']} ({job['status']}) with this suffix + dataset is already on the service : taking it (a new one : --suffix {bump(suffix)})")
	# the service took a job on it, so it is processed
	st.put(dataset_name = ds,dataset_id = st.get("dataset_id") if ds == st.get("dataset_name") else None,dataset_status = "processed",
		job_id = job["id"],job_status = job["status"],suffix = suffix,steps = steps,
		fine_tuned_model = job.get("fine_tuned_model"),t_job = now(),t_done = None)
	print(chat(job))
	return job

def jkey(job):
	p = job.get("progress") or {}
	return (job["status"],job.get("queue_position"),p.get("phase"),p.get("step"),job.get("fine_tuned_model"))

def waitJob(c,st,sleep = time.sleep,every = JOB_POLL,watch = True,last = None):
	ref = None
	while True:
		job = call(c.get_finetuning_status,st["job_id"],sleep = sleep,tries = 999)
		p = job.get("progress") or {}
		if ref is None and p.get("step"): ref = (ts(p.get("updated_at")) or time.time(),p["step"])
		if jkey(job) != last:
			last = jkey(job)
			print(chat(job,ref))
			st.put(job_status = job["status"],fine_tuned_model = job.get("fine_tuned_model"),suffix = job.get("suffix",st.get("suffix")))
		if job.get("fine_tuned_model"):
			if not st.get("t_done"): st.put(t_done = now())
			ready(job["fine_tuned_model"])
			return job
		if job["status"] in DEAD:
			raise Fail(f"job {job['id']} {job['status']} : {job.get('error')}\nfix it, then run the same command again to queue a new job")
		if not watch: return job
		sleep(every)

def run(c,st,path,name = "dreamhands-v1",emb = "yam-sim",suffix = "dreamhands",steps = 500,wait = True,sleep = time.sleep,**knobs):
	if tail(st.get("dataset_name")) != name or st.get("dataset_status") == "failed":
		check(c,path)
		embodiment(c,st,emb,sleep)
		upload(c,st,path,name,sleep)
	else:
		print(f"dataset {st['dataset_name']} already uploaded (state file), status : {st.get('dataset_status')}")
	last = None
	if st.get("job_id") and st.get("suffix") == suffix and st.get("job_status") not in DEAD:
		print(f"job {st['job_id']} already queued (state file) : polling")
	else:
		if st.get("dataset_status") != "processed": waitDataset(c,st,sleep)
		last = jkey(finetune(c,st,suffix,steps,sleep,**knobs))
	if wait: return waitJob(c,st,sleep,last = last)
	print(f"not waiting. follow it : {ME} status --watch")

def status(c,st,watch = False,job = None,sleep = time.sleep):
	if job: st.put(job_id = job,job_status = None,fine_tuned_model = None,t_done = None)
	if st.get("job_id"): return waitJob(c,st,sleep,watch = watch)
	if not st.get("dataset_id"): raise Fail(f"nothing in {st.path} yet : {ME} run {DIR}")
	if watch:
		waitDataset(c,st,sleep)
		return print(f"next : {ME} finetune")
	ds = call(c.get_dataset_status,st["dataset_id"],sleep = sleep)
	st.put(dataset_status = ds.status)
	print(f"dataset {ds.name} ({ds.id}) : {ds.status}" + (f" : {ds.error}" if ds.error else "") + " | no job yet")

def retry(c,st,sleep = time.sleep):
	if not st.get("dataset_id"): raise Fail(f"no dataset in the state file : {ME} run {DIR}")
	ds = call(c.retry_dataset,st["dataset_id"],sleep = sleep)
	st.put(dataset_status = ds.status)
	waitDataset(c,st,sleep)
	print(f"next : {ME} finetune")

def queue(c):
	q = getattr(c,"queue",fine_tuning.queue)()
	print(f"queue : {q.get('jobs_ahead_of_new')} ahead of a new job | running : {q.get('running')} | "
		f"slots free : {q.get('slots_free')}/{q.get('slots_total')} | at {q.get('updated_at')}")

def jobs(c):
	for j in getattr(c,"list",fine_tuning.list)():
		print(f"{j['id']} | {j.get('status')} | {j.get('dataset')} | suffix : {j.get('suffix')} | model : {j.get('fine_tuned_model')} | {j.get('created_at')}")


# ---- cli

def slug(n):
	def f(s):
		if not re.fullmatch(f"[a-z0-9-]{{1,{n}}}",s): raise argparse.ArgumentTypeError(f"use a-z, 0-9 and '-', {n} chars max")
		return s
	return f

def nsteps(s):
	n = int(s)
	if n % 100 or not 500 <= n <= 10000: raise argparse.ArgumentTypeError("steps : a multiple of 100 from 500 to 10000")
	return n

def parse(argv = None):
	p = argparse.ArgumentParser(prog = ME,description = __doc__)
	p.add_argument("--state",default = STATE,help = "state file (default data/faas.json)")
	sub = p.add_subparsers(dest = "cmd",required = True)
	def add(name,help,opts = ""):
		s = sub.add_parser(name,help = help)
		if "d" in opts: s.add_argument("dir",nargs = "?",default = DIR,help = "LeRobot session folder")
		if "n" in opts: s.add_argument("--name",type = slug(64),help = "dataset name (default : the state file's, else dreamhands-v1)")
		if "e" in opts: s.add_argument("--emb",type = slug(64),help = "embodiment name (default : the state file's, else yam-sim)")
		if "f" in opts:
			s.add_argument("--suffix",type = slug(40),help = "model id tail (default : the state file's, else dreamhands)")
			s.add_argument("--steps",default = 500,type = nsteps,help = "multiple of 100 from 500 to 10000 (default 500)")
			s.add_argument("--lr-schedule",choices = ("constant","cosine"))
			s.add_argument("--base-share",type = float,choices = (0.0,0.5))
			s.add_argument("--lr",type = float,dest = "learning_rate")
			s.add_argument("--resume-from",help = "one of our dream-1-ft:... models to keep training")
		if "w" in opts: s.add_argument("--no-wait",action = "store_true")
		return s
	add("models","list the models, also the 'is my key working' check")
	add("check","local dataset check, no key needed","d")
	add("run","check -> embodiment -> upload -> processed -> fine-tune -> poll, resumable","dnefw")
	s = add("status","job (or dataset) status from the state file, one line")
	s.add_argument("--watch",action = "store_true",help = "poll every 60 s until done")
	s.add_argument("--job",help = "follow this job id (ftj_...) from now on")
	add("queue","fine-tune queue numbers")
	add("jobs","all our fine-tune jobs")
	add("embodiment","register the YAM embodiment","e")
	add("upload","check + upload + wait until processed","dnew")
	add("finetune","queue a job on the uploaded dataset","fw").add_argument("--dataset",help = f"{USER}/<name> or just <name>, default : the state file's")
	add("retry","process a failed dataset again")
	return p.parse_args(argv)

def dispatch(c,st,a,sleep):
	knobs = {k: getattr(a,k,None) for k in ("lr_schedule","base_share","learning_rate","resume_from")}
	name = getattr(a,"name",None) or tail(st.get("dataset_name")) or "dreamhands-v1"
	emb = getattr(a,"emb",None) or tail(st.get("embodiment")) or "yam-sim"
	suffix = getattr(a,"suffix",None) or st.get("suffix") or "dreamhands"
	if a.cmd == "models": models(c,sleep)
	elif a.cmd == "check": check(c,a.dir)
	elif a.cmd == "run": run(c,st,a.dir,name,emb,suffix,a.steps,not a.no_wait,sleep,**knobs)
	elif a.cmd == "status": status(c,st,a.watch,a.job,sleep)
	elif a.cmd == "queue": queue(c)
	elif a.cmd == "jobs": jobs(c)
	elif a.cmd == "embodiment": embodiment(c,st,emb,sleep)
	elif a.cmd == "retry": retry(c,st,sleep)
	elif a.cmd == "upload":
		check(c,a.dir)
		embodiment(c,st,emb,sleep)
		upload(c,st,a.dir,name,sleep)
		if not a.no_wait: waitDataset(c,st,sleep)
	elif a.cmd == "finetune":
		job = finetune(c,st,suffix,a.steps,sleep,full(a.dataset),**knobs)
		if not a.no_wait: waitJob(c,st,sleep,last = jkey(job))

def main(argv = None,c = None,sleep = time.sleep):
	a = parse(argv)
	st = St(a.state)
	try:
		dispatch(c or client(a.cmd != "check"),st,a,sleep)
	except KeyboardInterrupt:
		print(f"\nstopped, nothing is lost (state : {st.path}). run the same command again, or just follow : {ME} status --watch")
		return 130
	except Fail as e:
		print("FAIL :",e)
		return 1
	except DreamError as e:
		print("FAIL :",explain(e))
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
