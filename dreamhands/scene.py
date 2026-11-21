"""Randomised episode starts. The stock apple sits on the workspace edge (x 0.15), so it is
shifted inward : a scripted pick succeeds on the whole box below, not beyond it."""
import numpy as np

LO = np.array([-0.09,-0.06])
HI = np.array([-0.02,0.03])


def scatter(sim,rng = None,body = "apple"):
	"""call right after sim.reset(); returns the xy offset (m)"""
	m = sim.model
	adr = m.jnt_qposadr[m.body_jntadr[m.body(body).id]]
	d = np.random.default_rng(rng).uniform(LO,HI)
	sim.data.qpos[adr:adr + 2] += d
	return d
