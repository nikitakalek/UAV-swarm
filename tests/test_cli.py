"""M2-6: командный интерфейс стенда — run (одно/много зёрен), compare, explain."""

import csv
import json

import pytest

pytest.importorskip("yaml")

from apps.cli import main  # noqa: E402


@pytest.fixture
def scen(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "name: clitest\nt_end: 3600\nworld: {size: [600, 600], targets: {count: 5}}\n"
        "agents: {count: 3}\nmission: {n_cells: 4}\n"
        "events: [{t: 30, type: agent_failure, agent: 1}]\n",
        encoding="utf-8",
    )
    return p


def test_run_single_seed_writes_csv_and_manifest(tmp_path, scen):
    assert main(["run", "--scenario", str(scen), "--seed", "1", "--out", str(tmp_path / "o"), "--quiet"]) == 0
    d = tmp_path / "o" / "clitest__stub_roundrobin" / "seed_001"
    for f in ("metrics.csv", "trajectory.csv", "events.jsonl", "run_manifest.json", "tasks.csv", "world.json"):
        assert (d / f).exists(), f
    man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["seed"] == 1 and len(man["config_hash"]) == 64


def test_twenty_seeds_one_command(tmp_path, scen):
    assert main(["run", "--scenario", str(scen), "--seeds", "1-20", "--out", str(tmp_path / "o"), "--quiet"]) == 0
    base = tmp_path / "o" / "clitest__stub_roundrobin"
    rows = list(csv.DictReader(open(base / "summary.csv", encoding="utf-8")))
    assert [int(r["seed"]) for r in rows] == list(range(1, 21))
    assert len(list(base.glob("seed_*/metrics.csv"))) == 20


def test_compare_identical_and_different(tmp_path, scen, capsys):
    o1, o2, o3 = (str(tmp_path / x) for x in ("a", "b", "c"))
    main(["run", "--scenario", str(scen), "--seeds", "1-3", "--out", o1, "--quiet"])
    main(["run", "--scenario", str(scen), "--seeds", "1-3", "--out", o2, "--quiet"])
    main(["run", "--scenario", str(scen), "--seeds", "1-3", "--out", o3, "--quiet", "--set", "agents.speed=7"])
    assert main(["compare", o1, o2, "--strict"]) == 0
    assert "идентичны" in capsys.readouterr().out
    assert main(["compare", o1, o3, "--strict"]) == 1


def test_explain_prints_reasons(tmp_path, scen, capsys):
    main(["run", "--scenario", str(scen), "--seed", "1", "--out", str(tmp_path / "o"), "--quiet"])
    d = tmp_path / "o" / "clitest__stub_roundrobin" / "seed_001"
    assert main(["explain", str(d)]) == 0
    out = capsys.readouterr().out
    assert "agent_failure" in out and "отказ агента 1" in out


def test_validate_reports_errors(tmp_path, scen, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("agents: {count: 2}\nevents: [{t: 1, type: agent_failure, agent: 5}]\n", encoding="utf-8")
    assert main(["validate", str(scen)]) == 0
    assert main(["validate", str(bad)]) == 1
