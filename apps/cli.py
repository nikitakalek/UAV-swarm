"""Командный интерфейс испытательного стенда (M2-6): ``swarmcore run | compare | validate | viz | explain``.

Примеры::

    swarmcore run --scenario scenarios/nominal.yaml --seed 1
    swarmcore run --scenario scenarios/failures.yaml --seeds 1-20 --jobs 4
    swarmcore run --scenario scenarios/nominal.yaml --seed 1 --set agents.count=10 --tag N10
    swarmcore compare runs/nominal__stub_roundrobin runs/failures__stub_roundrobin
    swarmcore viz runs/failures__stub_roundrobin/seed_001 --out failures.gif
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

NON_DETERMINISTIC_PREFIX = "wall_"


# --------------------------------------------------------------------------- сценарий
def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        out: list[int] = []
        for part in args.seeds.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out
    if args.n_seeds:
        return list(range(args.seed, args.seed + args.n_seeds))
    return [args.seed]


def _set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_scenario_with_overrides(path: str, overrides: list[str], allocator: str | None):
    import yaml

    from swarmsim.scenario import Scenario

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for ov in overrides or []:
        if "=" not in ov:
            raise SystemExit(f"--set ожидает key=value, получено: {ov}")
        k, v = ov.split("=", 1)
        _set_path(data, k.strip(), yaml.safe_load(v))
    if allocator:
        data.setdefault("allocator", {})["name"] = allocator
    return Scenario.model_validate(data)


def _run_one(job: tuple[str, int, str, bool]) -> dict[str, Any]:
    scenario_json, seed, outdir, parquet = job
    from swarmsim.scenario import Scenario
    from swarmsim.sim import run

    sc = Scenario.model_validate_json(scenario_json)
    t0 = time.perf_counter()
    res = run(sc, seed)
    res.metrics["wall_run_s"] = time.perf_counter() - t0
    res.save(outdir, parquet=parquet)
    return {"scenario": sc.name, "allocator": res.allocator, "seed": seed, **res.metrics,
            "metrics_hash": res.manifest["metrics_hash"]}


def cmd_run(args: argparse.Namespace) -> int:
    sc = load_scenario_with_overrides(args.scenario, args.set, args.allocator)
    seeds = _parse_seeds(args)
    group = f"{sc.name}{'__' + args.tag if args.tag else ''}__{sc.allocator.name.replace(':', '.')}"
    base = Path(args.out) / group
    jobs = [(sc.canonical_json(), s, str(base / f"seed_{s:03d}"), args.parquet) for s in seeds]
    t0 = time.perf_counter()
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_run_one, jobs))
    else:
        rows = [_run_one(j) for j in jobs]
    rows.sort(key=lambda r: r["seed"])
    base.mkdir(parents=True, exist_ok=True)
    _write_csv(base / "summary.csv", rows)
    batch = {"format": "swarmcore-batch", "scenario": sc.name, "config_hash": sc.config_hash(),
             "allocator": sc.allocator.name, "seeds": seeds, "overrides": args.set or [],
             "source": str(args.scenario), "runs": [f"seed_{s:03d}" for s in seeds]}
    with open(base / "batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    if not args.quiet:
        print(f"{len(rows)} прогон(ов) → {base}  ({time.perf_counter() - t0:.1f} с)")
        _print_summary(rows)
    return 0


def _print_summary(rows: list[dict[str, Any]]) -> None:
    keys = ["CR", "T_detect_90", "makespan", "E_total_Wh", "T_recovery_mean", "N_col", "N_dup", "N_lost", "valid"]
    print("seed  " + "  ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print(f"{r['seed']:<5} " + "  ".join(f"{_fmt(r.get(k)):>14}" for k in keys))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4g}"
    return str(v)


# --------------------------------------------------------------------------- compare
def _load_rows(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("metrics.csv"))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append({k: _num(v) for k, v in r.items()})
    if not rows:
        raise SystemExit(f"в {path} нет metrics.csv")
    return rows


def _num(v: str) -> Any:
    if v == "":
        return math.nan
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v


def _key(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        k = (r["scenario"], r["seed"])
        if k in out:
            k = (r["scenario"], r["allocator"], r["seed"])
        out[k] = r
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def cmd_compare(args: argparse.Namespace) -> int:
    A, B = _key(_load_rows(Path(args.a))), _key(_load_rows(Path(args.b)))
    common = sorted(set(A) & set(B), key=str)
    if not common:
        # разные сценарии: сравниваем по зерну
        A = {(r["seed"],): r for r in A.values()}
        B = {(r["seed"],): r for r in B.values()}
        common = sorted(set(A) & set(B))
    if not common:
        print("нет общих прогонов (scenario, seed) для сравнения")
        return 2
    metric_keys = [k for k, v in A[common[0]].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed"]
    if args.metrics:
        metric_keys = [k for k in args.metrics.split(",") if k in metric_keys]
    print(f"A: {args.a}\nB: {args.b}\nпарных прогонов: {len(common)}\n")
    print(f"{'метрика':<22}{'mean A':>14}{'mean B':>14}{'B − A':>14}{'Δ%':>9}{'≠':>5}")
    n_diff_total = 0
    for k in metric_keys:
        a = [A[c].get(k, math.nan) for c in common]
        b = [B[c].get(k, math.nan) for c in common]
        nd = sum(not _same(x, y) for x, y in zip(a, b))
        if not k.startswith(NON_DETERMINISTIC_PREFIX):
            n_diff_total += nd
        ma, mb = _nanmean(a), _nanmean(b)
        d = mb - ma
        pct = 100 * d / abs(ma) if ma and not math.isnan(ma) else math.nan
        print(f"{k:<22}{_fmt(ma):>14}{_fmt(mb):>14}{_fmt(d):>14}{_fmt(pct):>9}{nd:>5}")
    va = sum(bool(A[c].get("valid")) for c in common)
    vb = sum(bool(B[c].get("valid")) for c in common)
    print(f"\nvalid: A {va}/{len(common)}, B {vb}/{len(common)}")
    if args.strict:
        if n_diff_total:
            print(f"STRICT: найдено {n_diff_total} расхождений детерминированных метрик")
            return 1
        print("STRICT: детерминированные метрики идентичны")
    return 0


def _nanmean(xs: list[Any]) -> float:
    v = [float(x) for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else math.nan


# --------------------------------------------------------------------------- прочее
def cmd_validate(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    rc = 0
    for p in args.scenarios:
        try:
            sc = load_scenario_with_overrides(p, [], None)
            print(f"OK   {p}  ({sc.name}, N={sc.agents.count}, событий={len(sc.events)}, hash={sc.config_hash()[:12]})")
        except (ValidationError, ValueError) as e:
            rc = 1
            print(f"FAIL {p}\n{e}")
    return rc


def cmd_viz(args: argparse.Namespace) -> int:
    from apps.viz import animate, plot_summary

    run_dir = Path(args.run)
    if args.png:
        print(plot_summary(run_dir, args.png))
    if args.out or not args.png:
        print(animate(run_dir, args.out or str(run_dir / "animation.gif"), fps=args.fps, speed=args.speed))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from swarmsim.eventlog import EventLog

    log = EventLog.from_jsonl(Path(args.run) / "events.jsonl")
    if args.task:
        for line in log.explain_task(args.task):
            print(line)
        return 0
    for e in log.of_type("assign", "reassign", "unassign", "task_released"):
        root = log.root_cause(e.seq)
        if root.type == "mission_start" and e.type == "assign" and not args.all:
            continue
        print(f"t={e.t:8.1f} #{e.seq:<5} {e.type:<13} {e.data.get('task'):<16} "
              f"{str(e.data.get('from')):>4} → {str(e.data.get('to')):<4} {e.data.get('reason', '')}  "
              f"[корень: {root.type}#{root.seq} t={root.t:.1f}]")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from swarmsim.sim import _write_csv as w

    w(path, rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swarmcore", description="SwarmCore — испытательный стенд прогонов")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="прогон сценария (одно или много зёрен)")
    r.add_argument("--scenario", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--seeds", help="список/диапазоны зёрен, например 1-20 или 1,5,7")
    r.add_argument("--n-seeds", type=int, help="N зёрен начиная с --seed")
    r.add_argument("--allocator", help="переопределить allocator.name")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="переопределить поле сценария")
    r.add_argument("--tag", help="метка в имени папки результатов")
    r.add_argument("--out", default="runs")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--parquet", action="store_true", help="дополнительно trajectory.parquet (нужен pyarrow)")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="сравнить два прогона/пакета прогонов")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--metrics", help="список метрик через запятую")
    c.add_argument("--strict", action="store_true", help="код 1, если детерминированные метрики отличаются")
    c.set_defaults(func=cmd_compare)

    v = sub.add_parser("validate", help="проверить сценарии по схеме")
    v.add_argument("scenarios", nargs="+")
    v.set_defaults(func=cmd_validate)

    z = sub.add_parser("viz", help="анимация/сводная картинка прогона из его журналов")
    z.add_argument("run", help="папка прогона (seed_XXX)")
    z.add_argument("--out", help="файл анимации (.gif/.mp4)")
    z.add_argument("--png", help="сводная картинка (.png)")
    z.add_argument("--fps", type=int, default=15)
    z.add_argument("--speed", type=float, default=20.0, help="модельных секунд на секунду видео")
    z.set_defaults(func=cmd_viz)

    e = sub.add_parser("explain", help="почему переназначали задачи (журнал объяснений)")
    e.add_argument("run")
    e.add_argument("--task")
    e.add_argument("--all", action="store_true", help="включая первичные назначения")
    e.set_defaults(func=cmd_explain)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
