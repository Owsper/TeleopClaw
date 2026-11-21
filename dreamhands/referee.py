"""Geometric PLACE detector on MuJoCo state : object released and lying in / on the container.

Not the team success rule : that one (apple lifted and held) is dreamhands.trials.Referee, the evalrun default.
This one is for "put X into the bowl / on the plate" prompts : evalrun --referee place, or Referee(sim,task).check as a recorder referee.
"""
import mujoco
import numpy as np

OBJS = ("apple","orange","banana")
# from yam_env.build_xml : r = footprint radius, surf = support surface above the body origin
CONTS = {
	"bowl": {"r": 0.076,"surf": 0.010},	# inner rim radius, inner floor
	"plate": {"r": 0.069,"surf": 0.004},	# disc radius, disc top
}
MARGIN = 0.01	# centre must be this far inside the radius
HMAX = 0.045	# centre at most this high above the surface
VMAX = 0.05	# m/s


def parse(task):
	t = (task or "").lower()
	obj = next((o for o in OBJS if o in t),"apple")
	return obj,"plate" if "plate" in t else "bowl"


class Referee:
	def __init__(self,sim,task=""):
		self.sim = sim
		self.obj,self.cont = parse(task)
		m = sim.model
		self.oid = m.body(self.obj).id
		self.cid = m.body(self.cont).id
		self.dof = m.body_dofadr[self.oid]
		self.arm = [i for i in range(m.nbody) if m.body(i).name.startswith(("left_","right_"))]

	def info(self):
		m,d = self.sim.model,self.sim.data
		c = CONTS[self.cont]
		p = d.xipos[self.oid] - d.xpos[self.cid]
		bod = m.geom_bodyid[d.contact.geom]
		oth = bod[(bod == self.oid).any(1)]
		oth = oth[oth != self.oid]
		r = {
			"obj": self.obj,
			"cont": self.cont,
			"dxy": float(np.hypot(p[0],p[1])),
			"dz": float(p[2] - c["surf"]),
			"v": float(np.linalg.norm(d.qvel[self.dof:self.dof + 3])),
			"touch": bool((oth == self.cid).any()),
			"held": bool(np.isin(oth,self.arm).any()),
		}
		r["ok"] = bool(
			r["dxy"] <= c["r"] - MARGIN and 0 < r["dz"] <= HMAX and r["v"] <= VMAX
			and r["touch"] and not r["held"]
		)
		return r

	def check(self):
		return self.info()["ok"]

	__call__ = check

	# teleport the object's centre, for tests and the eval dry-run
	def place(self,xyz):
		m,d = self.sim.model,self.sim.data
		adr = m.jnt_qposadr[m.body_jntadr[self.oid]]
		d.qpos[adr:adr + 3] += np.asarray(xyz) - d.xipos[self.oid]
		d.qvel[self.dof:self.dof + 6] = 0
		mujoco.mj_forward(m,d)

	def drop(self,h=0.1):
		self.place(self.sim.data.xpos[self.cid] + [0,0,h])
