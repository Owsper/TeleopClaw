"""Eval + dataset report : python -m dreamhands.report data/eval.jsonl [--quality data/quality.jsonl] [--out data/report] -> report.md, chart.png, quality.png"""
import argparse
import json
import math
import statistics
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

Z = 1.959964
NAMES = {"base": "base","ft": "fine-tuned"}
COLORS = ["#2a78d6","#eb6834","#1baf7a","#eda100"]
INK,MUTED,GRID,BG = "#0b0b0b","#52514e","#e1e0d9","#fcfcfb"


def load(path):
	p = Path(path) if path else None
	if not p or not p.exists():
		return []
	out = []
	for line in p.read_text().splitlines():
		try:
			out.append(json.loads(line))
		except ValueError:
			pass
	return out


def rank(run):
	order = list(NAMES)
	return order.index(run) if run in order else len(order),run


def wilson(k,n):
	if not n:
		return 0.0,0.0
	p = k / n
	den = 1 + Z * Z / n
	mid = (p + Z * Z / (2 * n)) / den
	half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
	# exact at the ends, float noise there would give the chart a negative whisker
	return 0.0 if k == 0 else mid - half,1.0 if k == n else mid + half


def stats(rows):
	# one entry per (tag or model, task); errored episodes are not trials, a re-run replaces the older line
	eps,errs = {},{}
	for r in rows:
		g = (r.get("tag") or r.get("model") or "?",r.get("task") or "?")
		if r.get("error"):
			errs[g] = errs.get(g,0) + 1
		else:
			eps[(*g,r.get("model"),r.get("ep"))] = r
	groups = {}
	for key,r in eps.items():
		groups.setdefault(key[:2],[]).append(r)
	for g in errs:
		groups.setdefault(g,[])
	out = []
	for (run,task),rs in groups.items():
		n = len(rs)
		win = [r for r in rs if r.get("success")]
		rtt = [r["rtt_ms"] for r in rs if r.get("rtt_ms") is not None]
		lo,hi = wilson(len(win),n)
		out.append({
			"run": run,
			"task": task,
			"n": n,
			"k": len(win),
			"rate": len(win) / n if n else 0.0,
			"lo": lo,
			"hi": hi,
			"sim_s": statistics.mean(r["sim_s"] for r in win) if win else None,
			"rtt_ms": statistics.median(rtt) if rtt else None,
			"errors": errs.get((run,task),0),
		})
	out.sort(key=lambda s: (s["task"],rank(s["run"])))
	return out


def pct(x):
	return f"{x * 100:.1f} %"


def num(x,fmt="{:.1f}"):
	return "-" if x is None else fmt.format(x)


def evalMd(st):
	md = ["## Eval in sim",""]
	by = {(s["run"],s["task"]): s for s in st}
	for task in dict.fromkeys(s["task"] for s in st):
		b,f = by.get(("base",task)),by.get(("ft",task))
		if b and f and b["n"] and f["n"]:
			md.append(f"- {task} : base {b['k']}/{b['n']} ({pct(b['rate'])}) -> fine-tuned {f['k']}/{f['n']} ({pct(f['rate'])})")
	if md[-1]:
		md.append("")
	md += [
		"| run | task | n | successes | success rate | Wilson 95 % | mean sim s (successes) | median rtt ms | errors |",
		"|---|---|---|---|---|---|---|---|---|",
	]
	for s in st:
		md.append(
			f"| {NAMES.get(s['run'],s['run'])} | {s['task']} | {s['n']} | {s['k']} | {pct(s['rate'])} | {pct(s['lo'])} - {pct(s['hi'])}"
			f" | {num(s['sim_s'])} | {num(s['rtt_ms'],'{:.0f}')} | {s['errors']} |"
		)
	md += ["","n = episodes that ran to a verdict; errored episodes (network, bad reply) are counted apart.","","![success rate](chart.png)",""]
	return md


def style(ax):
	ax.set_facecolor(BG)
	for side in ("top","right","left"):
		ax.spines[side].set_visible(False)
	ax.spines["bottom"].set_color("#c3c2b7")
	ax.tick_params(colors=MUTED,labelsize=18,length=0)


def evalChart(st,path):
	tasks = list(dict.fromkeys(s["task"] for s in st))
	runs = sorted({s["run"] for s in st if s["n"]},key=rank)
	w = min(0.8 / len(runs),0.35)
	fig,ax = plt.subplots(figsize=(max(10,3.2 * len(tasks) + 3),6.8),facecolor=BG)
	style(ax)
	for i,run in enumerate(runs):
		ss = [s for s in st if s["run"] == run and s["n"]]
		xs = [tasks.index(s["task"]) + (i - (len(runs) - 1) / 2) * w for s in ss]
		ys = [s["rate"] * 100 for s in ss]
		lo = [y - s["lo"] * 100 for y,s in zip(ys,ss)]
		hi = [s["hi"] * 100 - y for y,s in zip(ys,ss)]
		ax.bar(xs,ys,w * 0.92,color=COLORS[i % len(COLORS)],label=NAMES.get(run,run),zorder=2)
		ax.errorbar(xs,ys,yerr=[lo,hi],fmt="none",ecolor=INK,capsize=7,elinewidth=2,capthick=2,zorder=3)
		for x,y,s in zip(xs,ys,ss):
			top = s["hi"] * 100 + 2.5
			ax.text(x,top,f"{y:.0f} %",ha="center",va="bottom",fontsize=22,fontweight="bold",color=INK)
			# over the label, not in the bar : a low whisker would run through it there
			ax.text(x,top + 10,f"{s['k']}/{s['n']}",ha="center",va="bottom",fontsize=17,color=MUTED)
	ax.set_xticks(range(len(tasks)))
	ax.set_xticklabels([textwrap.fill(t,20) for t in tasks],fontsize=19,color=INK)
	ax.set_xlim(-0.6,len(tasks) - 0.4)
	ax.set_ylim(0,126)
	ax.set_yticks([0,25,50,75,100])
	ax.set_ylabel("success rate (%)",fontsize=18,color=MUTED)
	ax.grid(axis="y",color=GRID,lw=1,zorder=0)
	ax.set_axisbelow(True)
	title = "Sim success : base vs fine-tuned" if {"base","ft"} <= set(runs) else "Sim success"
	ax.set_title(title,fontsize=26,fontweight="bold",color=INK,loc="left",pad=36)
	ax.text(0,1.03,"whiskers : Wilson 95 % interval, k/n : successes / episodes",transform=ax.transAxes,fontsize=14,color=MUTED)
	if len(runs) > 1:
		ax.legend(fontsize=18,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=len(runs),labelcolor=INK)
	fig.tight_layout()
	fig.savefig(path,dpi=150,facecolor=BG)
	plt.close(fig)


def quality(rows):
	why,tasks = {},{}
	for r in rows:
		t = tasks.setdefault(r.get("task") or "?",[0,0])
		t[0] += 1
		t[1] += bool(r.get("keep"))
		if not r.get("keep"):
			for k in dict.fromkeys(s.split(" : ")[0] for s in r.get("reasons") or ["no reason given"]):
				why[k] = why.get(k,0) + 1
	kept = sum(bool(r.get("keep")) for r in rows)
	return {"total": len(rows),"kept": kept,"dropped": len(rows) - kept,"why": dict(sorted(why.items(),key=lambda kv: -kv[1])),"tasks": tasks}


def qualityMd(q):
	md = [
		"## Dataset","",
		f"episodes recorded : {q['total']}, kept : {q['kept']}, dropped : {q['dropped']}"
		+ (f" (keep rate : {pct(q['kept'] / q['total'])})" if q["total"] else ""),"",
		"| task | recorded | kept | dropped |","|---|---|---|---|",
	]
	md += [f"| {t} | {n} | {k} | {n - k} |" for t,(n,k) in q["tasks"].items()]
	if q["why"]:
		md += ["","| drop reason | episodes |","|---|---|"]
		md += [f"| {k} | {n} |" for k,n in q["why"].items()]
		md += ["","An episode can be dropped for several reasons, so these can add up to more than the dropped count."]
	md += ["","![dataset quality](quality.png)",""]
	return md


def qualityChart(q,path):
	labels = ["kept"] + [f"dropped : {k}" for k in q["why"]]
	vals = [q["kept"]] + list(q["why"].values())
	fig,ax = plt.subplots(figsize=(10,1.6 + 0.85 * len(labels)),facecolor=BG)
	style(ax)
	ax.barh(range(len(vals)),vals,0.7,color=[COLORS[0]] + [COLORS[1]] * len(q["why"]),zorder=2)
	for i,v in enumerate(vals):
		ax.text(v + max(vals + [1]) * 0.015,i,str(v),va="center",fontsize=20,fontweight="bold",color=INK)
	ax.set_yticks(range(len(labels)))
	ax.set_yticklabels(labels,fontsize=19,color=INK)
	ax.invert_yaxis()
	ax.set_xlim(0,max(vals + [1]) * 1.12)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_xlabel("episodes",fontsize=17,color=MUTED)
	ax.grid(axis="x",color=GRID,lw=1,zorder=0)
	ax.set_axisbelow(True)
	ax.set_title(f"Recorded demos : {q['kept']} of {q['total']} kept",fontsize=24,fontweight="bold",color=INK,loc="left",pad=16)
	fig.tight_layout()
	fig.savefig(path,dpi=150,facecolor=BG)
	plt.close(fig)


def build(evalPath=None,qualityPath=None,out="data/report"):
	out = Path(out)
	out.mkdir(parents=True,exist_ok=True)
	md = ["# DreamHands report",""]
	st = stats(load(evalPath))
	if st:
		md += evalMd(st)
		if any(s["n"] for s in st):
			evalChart(st,out / "chart.png")
	qrows = load(qualityPath)
	if qrows:
		q = quality(qrows)
		md += qualityMd(q)
		qualityChart(q,out / "quality.png")
	if not st and not qrows:
		md.append("no eval or quality lines found.")
	(out / "report.md").write_text("\n".join(md) + "\n",encoding="utf-8")
	return out / "report.md"


def main(argv=None):
	p = argparse.ArgumentParser(description="eval + dataset report")
	p.add_argument("eval",nargs="?",default="data/eval.jsonl",help="evalrun jsonl (may be missing)")
	p.add_argument("--quality",help="quality.py jsonl")
	p.add_argument("--out",default="data/report")
	a = p.parse_args(argv)
	path = build(a.eval,a.quality,a.out)
	print(path.read_text(encoding="utf-8"))
	print(f"written : {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
