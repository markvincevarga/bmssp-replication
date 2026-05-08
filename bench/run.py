#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tomllib import load as load_toml

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TaskID,
    ProgressColumn,
)
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRITERION_DIR = PROJECT_ROOT / "target" / "criterion"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

console = Console()


@dataclass
class Source:
    spec: str
    label: str
    max_size: int | None = None
    sizes: list[int] | None = None  # if set, overrides config.sizes for this source

@dataclass
class RgSource:
    size: int
    spec: str
    label: str

@dataclass
class CriterionTier:
    max_size: int
    warm_up_ms: int
    measurement_s: int
    sample_size: int

@dataclass
class Config:
    algorithms: list[str]
    sizes: list[int]
    sources: list[Source]
    rg_sources: list[RgSource]
    tiers: list[CriterionTier]
    graph_instances: int = 1
    base_seed: int = 42

def spec_to_cache_key(spec: str) -> str:
    s = spec.strip()
    aliases = {
        "grid": "grid",
        "castro:d3": "castro_d3", "d3": "castro_d3",
        "castro:h3": "castro_h3", "h3": "castro_h3",
        "castro:sgrided": "castro_sgrided", "sgrided": "castro_sgrided",
        "castro:rgrided": "castro_rgrided", "rgrided": "castro_rgrided",
        "castro:sgridr": "castro_sgridr", "sgridr": "castro_sgridr",
        "castro:rgridr": "castro_rgridr", "rgridr": "castro_rgridr",
    }
    if s in aliases:
        return aliases[s]
    if ":" not in s:
        return s
    name, params = s.split(":", 1)
    if name == "fixed":
        ef = _parse_param(params, "ef")
        return f"ef{int(ef)}"
    if name == "ba":
        m = _parse_param(params, "m")
        return f"ba_m{int(m)}"
    if name == "ws":
        k = int(_parse_param(params, "k"))
        beta = float(_parse_param(params, "beta"))
        return f"ws_k{k}_b{int(beta * 100)}"
    if name == "rg":
        r = float(_parse_param(params, "radius"))
        return f"rg_r{int(r * 1000)}"
    if name == "dimacs":
        nm = _parse_param(params, "name")
        return "dimacs_" + "".join(
            c if (c.isascii() and (c.isalnum() or c in "_-")) else "_" for c in nm
        )
    return s


def _parse_param(params: str, key: str) -> str:
    for piece in re.split(r"[,:]", params):
        if "=" in piece:
            k, v = piece.split("=", 1)
            if k.strip() == key:
                return v.strip()
    raise ValueError(f"missing param '{key}' in '{params}'")


@dataclass
class Combo:
    algo: str
    size: int
    spec: str
    label: str
    graph_index: int = 0

    @property
    def key(self) -> str:
        return f"{self.algo}|{self.size}|{self.label}|g{self.graph_index}"

    @property
    def cache_key(self) -> str:
        return spec_to_cache_key(self.spec)

@dataclass
class RunState:
    completed: set[str] = field(default_factory=set)
    timings: dict[str, float] = field(default_factory=dict)

@dataclass
class Batch:
    algo: str
    sizes: list[int]
    specs: list[str]
    combos: list[Combo]


def load_config(path: Path) -> Config:
    with open(path, "rb") as f:
        raw = load_toml(f)

    tiers_raw = raw.get("criterion", {}).get("tiers", {})
    tiers = []
    for name in ("small", "medium", "large"):
        t = tiers_raw.get(name, {})
        tiers.append(CriterionTier(
            max_size=t.get("max_size", 999_999_999),
            warm_up_ms=t.get("warm_up_ms", 3000),
            measurement_s=t.get("measurement_s", 5),
            sample_size=t.get("sample_size", 100),
        ))
    tiers.sort(key=lambda t: t.max_size)

    matrix = raw.get("matrix", {})
    algorithms = matrix.get("algorithms", [])
    sizes = matrix.get("sizes", [])

    if not algorithms:
        console.print("[red]No algorithms defined in config[/red]")
        raise SystemExit(1)
    if not sizes:
        console.print("[red]No sizes defined in config[/red]")
        raise SystemExit(1)

    sources = []
    for s in matrix.get("sources", []):
        sizes_override = s.get("sizes")
        sources.append(Source(
            spec=s["spec"],
            label=s["label"],
            max_size=s.get("max_size"),
            sizes=sorted(sizes_override) if sizes_override else None,
        ))

    rg_sources = []
    for r in matrix.get("rg_sources", []):
        rg_sources.append(RgSource(size=r["size"], spec=r["spec"], label=r["label"]))

    graph_cfg = raw.get("graph", {})
    graph_instances = max(1, graph_cfg.get("instances", 1))
    base_seed = graph_cfg.get("base_seed", 42)

    return Config(
        algorithms=algorithms,
        sizes=sorted(sizes),
        sources=sources,
        rg_sources=rg_sources,
        tiers=tiers,
        graph_instances=graph_instances,
        base_seed=base_seed,
    )


def sizes_for_source(config: Config, src: Source) -> list[int]:
    if src.sizes is not None:
        return src.sizes
    if src.max_size is not None:
        return [s for s in config.sizes if s <= src.max_size]
    return list(config.sizes)


def build_matrix(config: Config) -> list[Combo]:
    combos: list[Combo] = []
    for algo in config.algorithms:
        for src in config.sources:
            for size in sizes_for_source(config, src):
                for gi in range(config.graph_instances):
                    combos.append(Combo(algo=algo, size=size, spec=src.spec, label=src.label, graph_index=gi))
        for size in config.sizes:
            for rg in config.rg_sources:
                if rg.size == size:
                    for gi in range(config.graph_instances):
                        combos.append(Combo(algo=algo, size=size, spec=rg.spec, label=rg.label, graph_index=gi))
    return combos


def build_missing(config: Config) -> list[dict]:
    missing = []
    for algo in config.algorithms:
        for src in config.sources:
            allowed = set(sizes_for_source(config, src))
            for size in config.sizes:
                if size not in allowed:
                    missing.append({
                        "algo": algo,
                        "size": size,
                        "source": src.label,
                        "reason": "excluded_by_config",
                    })
    return missing


def scan_completed(combos: list[Combo], criterion_dir: Path, graph_instances: int = 1) -> RunState:
    state = RunState()
    for c in combos:
        has_estimates = criterion_estimates_path(c, criterion_dir, graph_instances).exists()
        has_memory = criterion_memory_path(c, criterion_dir, graph_instances).exists()
        if has_estimates and has_memory:
            state.completed.add(c.key)
    return state


def combo_dir(c: Combo, criterion_dir: Path, graph_instances: int = 1) -> Path:
    base = criterion_dir / f"{c.algo}_comprehensive" / "comprehensive"
    key = c.cache_key
    if graph_instances > 1:
        multi = base / f"size_{c.size}_{key}_g{c.graph_index}"
        if c.graph_index == 0 and not multi.exists():
            legacy = base / f"size_{c.size}_{key}"
            if legacy.exists():
                return legacy
        return multi
    return base / f"size_{c.size}_{key}"


def criterion_estimates_path(c: Combo, criterion_dir: Path, graph_instances: int = 1) -> Path:
    return combo_dir(c, criterion_dir, graph_instances) / "new" / "estimates.json"


def criterion_sample_path(c: Combo, criterion_dir: Path, graph_instances: int = 1) -> Path:
    return combo_dir(c, criterion_dir, graph_instances) / "new" / "sample.json"


def criterion_memory_path(c: Combo, criterion_dir: Path, graph_instances: int = 1) -> Path:
    return combo_dir(c, criterion_dir, graph_instances) / "memory.json"


def detect_hardware() -> dict:
    cpu = "unknown"
    ram_bytes = 0
    sys_name = platform.system()

    if sys_name == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            console.print("[yellow]Warning: could not detect CPU[/yellow]")
        try:
            ram_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            console.print("[yellow]Warning: could not detect RAM[/yellow]")
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            console.print("[yellow]Warning: could not detect CPU[/yellow]")
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        ram_bytes = kb * 1024
                        break
        except (OSError, ValueError):
            console.print("[yellow]Warning: could not detect RAM[/yellow]")

    arch = platform.machine()
    os_version = f"{sys_name} {platform.release()}"
    ram_gb = round(ram_bytes / (1024 ** 3))

    hw_string = f"{cpu}|{ram_bytes}|{arch}"
    hardware_id = hashlib.sha256(hw_string.encode()).hexdigest()[:16]

    return {
        "cpu": cpu,
        "arch": arch,
        "ram_gb": ram_gb,
        "os": os_version,
        "hardware_id": hardware_id,
    }


def detect_git() -> dict:
    info: dict = {"commit": None, "branch": None, "dirty": None}
    repo_root = Path(__file__).resolve().parent.parent
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        )
        info["dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        console.print("[yellow]Warning: could not detect git commit[/yellow]")
    return info


def tier_for_size(size: int, tiers: list[CriterionTier]) -> CriterionTier:
    for t in tiers:
        if size <= t.max_size:
            return t
    return tiers[-1]


def estimate_combo_seconds(size: int, tiers: list[CriterionTier]) -> float:
    t = tier_for_size(size, tiers)
    return (t.warm_up_ms / 1000.0) + t.measurement_s + 2.0


def estimate_total(pending: list[Combo], tiers: list[CriterionTier], state: RunState) -> float:
    total = 0.0
    for c in pending:
        observed_key = f"tier_{tier_for_size(c.size, tiers).max_size}"
        if observed_key in state.timings:
            total += state.timings[observed_key]
        else:
            total += estimate_combo_seconds(c.size, tiers)
    return total


def collect_combo_result(c: Combo, criterion_dir: Path, graph_instances: int = 1) -> dict | None:
    est_path = criterion_estimates_path(c, criterion_dir, graph_instances)
    if not est_path.exists():
        return None

    try:
        with open(est_path, encoding="utf-8") as f:
            estimates = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[yellow]Warning: bad estimates for {c.key}: {e}[/yellow]")
        return None

    result: dict = {"runtime_ns": estimates}

    sample_path = criterion_sample_path(c, criterion_dir, graph_instances)
    if sample_path.exists():
        try:
            with open(sample_path, encoding="utf-8") as f:
                result["samples"] = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    mem_path = criterion_memory_path(c, criterion_dir, graph_instances)
    if mem_path.exists():
        try:
            with open(mem_path, encoding="utf-8") as f:
                mem = json.load(f)
            result["memory_bytes"] = mem.get("peak_bytes")
        except (json.JSONDecodeError, OSError):
            pass

    return result


def _aggregate_instances(instance_results: list[dict]) -> dict:
    n = len(instance_results)
    if n == 1:
        return instance_results[0]

    per_means = []
    for r in instance_results:
        try:
            per_means.append(r["runtime_ns"]["mean"]["point_estimate"])
        except (KeyError, TypeError):
            pass

    if not per_means:
        return instance_results[0]

    grand_mean = sum(per_means) / len(per_means)
    if len(per_means) > 1:
        between_var = sum((m - grand_mean) ** 2 for m in per_means) / (len(per_means) - 1)
    else:
        between_var = 0.0
    between_stddev = between_var ** 0.5
    se = between_stddev / (len(per_means) ** 0.5)

    within_stddevs = []
    for r in instance_results:
        try:
            within_stddevs.append(r["runtime_ns"]["std_dev"]["point_estimate"])
        except (KeyError, TypeError):
            pass
    mean_within_stddev = (sum(s for s in within_stddevs) / len(within_stddevs)) if within_stddevs else 0.0

    combined_stddev = (between_var + mean_within_stddev ** 2) ** 0.5

    runtime_ns = {
        "mean": {
            "point_estimate": grand_mean,
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": grand_mean - 1.96 * se,
                "upper_bound": grand_mean + 1.96 * se,
            },
            "standard_error": se,
        },
        "std_dev": {
            "point_estimate": combined_stddev,
        },
        "graph_variance": {
            "between_graphs_stddev": between_stddev,
            "mean_within_graph_stddev": mean_within_stddev,
            "combined_stddev": combined_stddev,
            "per_instance_means": per_means,
        },
    }

    mem_vals = [r["memory_bytes"] for r in instance_results if r.get("memory_bytes") is not None]
    if mem_vals:
        mem_mean = sum(mem_vals) / len(mem_vals)
        if len(mem_vals) > 1:
            mem_var = sum((v - mem_mean) ** 2 for v in mem_vals) / (len(mem_vals) - 1)
        else:
            mem_var = 0.0
        mem_stddev = mem_var ** 0.5
        memory_bytes = round(mem_mean)
        memory_stats = {
            "mean": mem_mean,
            "stddev": mem_stddev,
            "min": min(mem_vals),
            "max": max(mem_vals),
            "per_instance": mem_vals,
        }
    else:
        memory_bytes = None
        memory_stats = None

    result: dict = {
        "graph_instances": n,
        "runtime_ns": runtime_ns,
        "per_instance": instance_results,
    }
    if memory_bytes is not None:
        result["memory_bytes"] = memory_bytes
        result["memory_stats"] = memory_stats

    return result


def collect_all_results(combos: list[Combo], criterion_dir: Path, graph_instances: int = 1) -> dict:
    by_group: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for c in combos:
        data = collect_combo_result(c, criterion_dir, graph_instances)
        if data is None:
            continue
        by_group[(c.algo, c.label, c.size)].append(data)

    results: dict = {}
    for (algo, label, size), instance_data in by_group.items():
        aggregated = _aggregate_instances(instance_data)
        results.setdefault(algo, {}).setdefault(label, {})[str(size)] = aggregated
    return results


def plan_batches(combos: list[Combo]) -> list[Batch]:
    by_algo: dict[str, list[Combo]] = defaultdict(list)
    for c in combos:
        by_algo[c.algo].append(c)

    batches: list[Batch] = []
    for algo, algo_combos in by_algo.items():
        spec_to_sizes: dict[str, set[int]] = defaultdict(set)
        for c in algo_combos:
            spec_to_sizes[c.spec].add(c.size)

        sizeset_to_specs: dict[frozenset[int], list[str]] = defaultdict(list)
        for spec, sizes in spec_to_sizes.items():
            sizeset_to_specs[frozenset(sizes)].append(spec)

        for size_set, specs in sizeset_to_specs.items():
            batch_combos = [
                c for c in algo_combos
                if c.spec in specs and c.size in size_set
            ]
            batches.append(Batch(
                algo=algo,
                sizes=sorted(size_set),
                specs=specs,
                combos=batch_combos,
            ))

    return batches


BENCH_RE = re.compile(r"Benchmarking (\S+?)(?::\s|$)")
SIZE_SOURCE_RE = re.compile(r"size_(\d+)_(.+?)(?:_g(\d+))?$")


class ETAColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.total_estimate: float | None = None

    def set_estimate(self, elapsed: float, remaining: float) -> None:
        self.total_estimate = elapsed + remaining

    def render(self, task) -> Text:
        if self.total_estimate is None:
            return Text("-:--:--", style="progress.remaining")
        h, rem = divmod(int(self.total_estimate), 3600)
        m, s = divmod(rem, 60)
        return Text(f"{h}:{m:02d}:{s:02d}", style="progress.remaining")


def run_batch(
    batch: Batch,
    state: RunState,
    config: Config,
    criterion_dir: Path,
    progress: Progress,
    task_id: TaskID,
    all_pending: list[Combo],
    eta_col: ETAColumn,
    algo_label: str,
) -> bool:
    env = os.environ.copy()
    env["BENCH_ALGORITHMS"] = batch.algo
    env["BENCH_SIZES"] = ",".join(str(s) for s in batch.sizes)
    env["BENCH_GRAPH_SOURCES"] = ",".join(batch.specs)
    env["BENCH_GRAPH_INSTANCES"] = str(config.graph_instances)
    env["BENCH_BASE_SEED"] = str(config.base_seed)

    cmd = [
        "cargo", "bench",
        "--bench", "algorithms",
        "--features", "track-alloc",
    ]

    seen_tags: set[str] = set()
    batch_total = len(batch.combos)

    progress.update(task_id, description=f"[dim]{algo_label} compiling...")

    batch_start = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    stderr_lines: list[str] = []
    assert proc.stderr is not None
    try:
        for raw_line in proc.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            stderr_lines.append(line)
            m = BENCH_RE.search(line)
            if m:
                bench_id = m.group(1)
                parts = bench_id.rsplit("/", 1)
                tag = parts[-1] if len(parts) > 1 else bench_id
                if tag in seen_tags:
                    continue
                seen_tags.add(tag)
                sm = SIZE_SOURCE_RE.match(tag)
                if sm:
                    n, src = sm.group(1), sm.group(2)
                    progress.update(task_id, description=f"{algo_label} n={n} {src} [{len(seen_tags)}/{batch_total}]")
                else:
                    progress.update(task_id, description=f"{algo_label} {tag} [{len(seen_tags)}/{batch_total}]")

        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise

    batch_elapsed = time.monotonic() - batch_start

    if proc.returncode != 0:
        console.print(f"[red]cargo bench failed for {batch.algo}[/red]")
        console.print("\n".join(stderr_lines[-40:]))
        return False

    newly_completed = 0
    gi = config.graph_instances
    for c in batch.combos:
        if c.key not in state.completed and criterion_estimates_path(c, criterion_dir, gi).exists() and criterion_memory_path(c, criterion_dir, gi).exists():
            state.completed.add(c.key)
            newly_completed += 1

    if newly_completed > 0:
        per_combo = batch_elapsed / newly_completed
        for c in batch.combos:
            tier_key = f"tier_{tier_for_size(c.size, config.tiers).max_size}"
            state.timings[tier_key] = per_combo

    progress.advance(task_id, newly_completed)

    still_pending = [c for c in all_pending if c.key not in state.completed]
    task_elapsed = progress.tasks[task_id].elapsed or 0.0  # pylint: disable=invalid-sequence-index
    eta_col.set_estimate(task_elapsed, estimate_total(still_pending, config.tiers, state))

    return True


def dry_run(config: Config, combos: list[Combo], state: RunState) -> None:
    pending = [c for c in combos if c.key not in state.completed]
    completed = [c for c in combos if c.key in state.completed]

    table = Table(title="Benchmark Matrix")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Sizes")
    table.add_column("Sources")
    table.add_column("Combos", justify="right")
    table.add_column("Pending", justify="right")

    by_algo = defaultdict(list)
    for c in combos:
        by_algo[c.algo].append(c)

    pending_set = {c.key for c in pending}
    for algo in config.algorithms:
        algo_combos = by_algo[algo]
        algo_pending = [c for c in algo_combos if c.key in pending_set]
        sizes = sorted({c.size for c in algo_combos})
        table.add_row(
            algo,
            f"{len(sizes)} ({min(sizes):,}-{max(sizes):,})",
            str(len({c.label for c in algo_combos})),
            str(len(algo_combos)),
            str(len(algo_pending)),
        )

    console.print(table)
    console.print()

    est = estimate_total(pending, config.tiers, state)
    console.print(f"[bold]Total combos:[/bold] {len(combos)}")
    console.print(f"[bold]Completed:[/bold] {len(completed)}")
    console.print(f"[bold]Pending:[/bold] {len(pending)}")
    console.print(f"[bold]Estimated time:[/bold] {format_duration(est)}")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runner for duan SSSP algorithms")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "bench_config.toml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--graph-instances", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.graph_instances is not None:
        if args.graph_instances < 1:
            console.print("[red]--graph-instances must be >= 1[/red]")
            raise SystemExit(1)
        config.graph_instances = args.graph_instances

    if args.algorithms:
        algo_filter = [a.strip() for a in args.algorithms.split(",")]
        unknown = [a for a in algo_filter if a not in config.algorithms]
        if unknown:
            console.print(f"[red]Unknown algorithms: {', '.join(unknown)}[/red]")
            console.print(f"[dim]Available: {', '.join(config.algorithms)}[/dim]")
            raise SystemExit(1)
        config.algorithms = [a for a in config.algorithms if a in algo_filter]

    if args.max_size:
        config.sizes = [s for s in config.sizes if s <= args.max_size]
        for src in config.sources:
            if src.sizes is not None:
                src.sizes = [s for s in src.sizes if s <= args.max_size]
        if not config.sizes:
            console.print(f"[red]No sizes <= {args.max_size}[/red]")
            raise SystemExit(1)

    combos = build_matrix(config)
    missing = build_missing(config)

    if args.fresh:
        state = RunState()
    else:
        state = scan_completed(combos, CRITERION_DIR, config.graph_instances)

    if args.dry_run:
        dry_run(config, combos, state)
        return

    pending = [c for c in combos if c.key not in state.completed]
    failed_batches: list[str] = []

    if not pending:
        console.print("[green]All combos already completed.[/green]")
    else:
        batches = plan_batches(pending)
        algo_order = [a for a in config.algorithms if any(b.algo == a for b in batches)]

        batches_sorted = []
        for a in algo_order:
            batches_sorted.extend(b for b in batches if b.algo == a)

        eta_col = ETAColumn()
        eta_col.set_estimate(0.0, estimate_total(pending, config.tiers, state))

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                eta_col,
                console=console,
            ) as progress:
                already_done = len(combos) - len(pending)
                task_id = progress.add_task("starting...", total=len(combos), completed=already_done)

                algo_idx: dict[str, int] = {}
                for b in batches_sorted:
                    if b.algo not in algo_idx:
                        algo_idx[b.algo] = len(algo_idx) + 1
                    algo_label = f"{b.algo} [{algo_idx[b.algo]}/{len(algo_order)}]"
                    ok = run_batch(b, state, config, CRITERION_DIR, progress, task_id, pending, eta_col, algo_label)
                    if not ok:
                        failed_batches.append(b.algo)

                total = progress.tasks[task_id].total  # pylint: disable=invalid-sequence-index
                if failed_batches:
                    progress.update(task_id, description=f"[yellow]done with {len(failed_batches)} failures", completed=total)
                else:
                    progress.update(task_id, description="[green]done", completed=total)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Saving partial results.[/yellow]")

    if failed_batches:
        console.print(f"[yellow]Failed batches: {', '.join(failed_batches)}[/yellow]")

    hardware = detect_hardware()
    git_info = detect_git()
    results = collect_all_results(combos, CRITERION_DIR, config.graph_instances)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    output = {
        "graph": {"instances": config.graph_instances, "base_seed": config.base_seed},
        "hardware": hardware,
        "git": git_info,
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "results": results,
        "missing": missing,
    }

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"run_{run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        console.print(f"\n[green]Results written to {out_path}[/green]")
    except OSError as e:
        console.print(f"[red]Failed to write results: {e}[/red]")
        console.print("[dim]Raw data is still in target/criterion/[/dim]")
        raise SystemExit(1) from e

    combo_count = sum(
        len(size_data)
        for algo_data in results.values()
        for size_data in algo_data.values()
    )
    console.print(f"[dim]{combo_count} data points across {len(results)} algorithms[/dim]")


if __name__ == "__main__":
    main()
