import json

import pytest

from dreamhands import report

PICK,BOWL = "pick up the apple","put apple into the bowl"


def ep(tag,task,i,ok,sim_s=20.0,rtt=800,error=None):
	return {
		"model": f"m/{tag}","tag": tag,"task": task,"ep": i,"success": ok,"sim_s": sim_s,
		"wall_s": 30.0,"calls": 9,"rtt_ms": rtt,"error": error,"reason": "x","referee": "lift",
	}


def write(path,rows):
	path.write_text("".join(json.dumps(r) + "\n" for r in rows))
	return path


@pytest.fixture
def evalFile(tmp_path):
	rows = [ep("base",PICK,i,i < 3,sim_s=40.0) for i in range(10)]
	# ft : 7/10, successes at 10 s and 20 s alternating, rtt 600 except one 1000
	rows += [ep("ft",PICK,i,i < 7,sim_s=10.0 if i % 2 else 20.0,rtt=1000 if i == 0 else 600) for i in range(10)]
	rows += [ep("ft",BOWL,i,i < 1) for i in range(4)]
	# an errored episode then its re-run : one trial, one error
	rows += [ep("base",BOWL,0,False,error="DreamTimeout : no answer"),ep("base",BOWL,0,False),ep("base",BOWL,1,False)]
	return write(tmp_path / "eval.jsonl",rows)


@pytest.fixture
def qualityFile(tmp_path):
	rows = [{"idx": i,"keep": True,"reasons": [],"stats": {},"task": PICK} for i in range(5)]
	rows += [
		{"idx": 5,"keep": False,"reasons": ["track loss : 8.0 % >= 5 %"],"stats": {},"task": PICK},
		{"idx": 6,"keep": False,"reasons": ["track loss : 12.5 % >= 5 %","dur : 1.20 s < 1.6 s"],"stats": {},"task": PICK},
		{"idx": 7,"keep": False,"reasons": ["referee : fail"],"stats": {},"task": BOWL},
	]
	return write(tmp_path / "quality.jsonl",rows)


def test_wilson_by_hand():
	# k = 7, n = 10, z = 1.96 : centre (0.7 + 0.19208) / 1.38416 = 0.6445, half 1.96 * sqrt(0.030604) / 1.38416 = 0.2477
	lo,hi = report.wilson(7,10)
	assert lo == pytest.approx(0.3968,abs=5e-4) and hi == pytest.approx(0.8922,abs=5e-4)
	lo,hi = report.wilson(0,10)
	assert lo == 0 and hi == pytest.approx(0.2775,abs=5e-4)
	assert report.wilson(0,0) == (0.0,0.0)


def test_stats(evalFile):
	st = {(s["run"],s["task"]): s for s in report.stats(report.load(evalFile))}
	ft = st[("ft",PICK)]
	assert (ft["n"],ft["k"],ft["rate"]) == (10,7,0.7)
	assert ft["sim_s"] == pytest.approx((20 * 4 + 10 * 3) / 7)
	assert ft["rtt_ms"] == 600
	b = st[("base",BOWL)]
	assert (b["n"],b["k"],b["errors"]) == (2,0,1)


def test_eval_only(evalFile,tmp_path):
	out = tmp_path / "rep"
	md = report.build(evalFile,None,out).read_text(encoding="utf-8")
	assert f"| fine-tuned | {PICK} | 10 | 7 | 70.0 % | 39.7 % - 89.2 % | 15.7 | 600 | 0 |" in md
	assert f"| base | {PICK} | 10 | 3 | 30.0 % | 10.8 % - 60.3 % | 40.0 | 800 | 0 |" in md
	assert f"| fine-tuned | {BOWL} | 4 | 1 | 25.0 % |" in md
	assert f"| base | {BOWL} | 2 | 0 | 0.0 % | 0.0 % - 65.8 % | - | 800 | 1 |" in md
	assert f"- {PICK} : base 3/10 (30.0 %) -> fine-tuned 7/10 (70.0 %)" in md
	assert "## Dataset" not in md
	assert (out / "chart.png").stat().st_size > 5000
	assert not (out / "quality.png").exists()


def test_quality_only(qualityFile,tmp_path):
	out = tmp_path / "rep"
	md = report.build(tmp_path / "missing.jsonl",qualityFile,out).read_text(encoding="utf-8")
	assert "episodes recorded : 8, kept : 5, dropped : 3 (keep rate : 62.5 %)" in md
	assert "| track loss | 2 |" in md and "| dur | 1 |" in md and "| referee | 1 |" in md
	assert f"| {PICK} | 7 | 5 | 2 |" in md and f"| {BOWL} | 1 | 0 | 1 |" in md
	assert "## Eval" not in md
	assert (out / "quality.png").stat().st_size > 5000
	assert not (out / "chart.png").exists()


def test_both_and_cli(evalFile,qualityFile,tmp_path,capsys):
	out = tmp_path / "rep"
	assert report.main([str(evalFile),"--quality",str(qualityFile),"--out",str(out)]) == 0
	md = (out / "report.md").read_text(encoding="utf-8")
	assert "## Eval in sim" in md and "## Dataset" in md
	assert "![success rate](chart.png)" in md and "![dataset quality](quality.png)" in md
	assert (out / "chart.png").stat().st_size > 5000 and (out / "quality.png").stat().st_size > 5000
	assert "70.0 %" in capsys.readouterr().out


def test_all_or_nothing_rates(tmp_path):
	rows = [ep("base",PICK,i,False) for i in range(10)] + [ep("ft",PICK,i,True) for i in range(10)]
	out = tmp_path / "rep"
	md = report.build(write(tmp_path / "e.jsonl",rows),None,out).read_text(encoding="utf-8")
	assert "| 10 | 0 | 0.0 % | 0.0 % - 27.8 % |" in md and "| 10 | 10 | 100.0 % | 72.2 % - 100.0 % |" in md
	assert (out / "chart.png").stat().st_size > 5000


def test_nothing(tmp_path):
	md = report.build(tmp_path / "no.jsonl",None,tmp_path / "rep").read_text(encoding="utf-8")
	assert "no eval or quality lines found" in md
