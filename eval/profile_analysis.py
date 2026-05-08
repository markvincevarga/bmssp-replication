#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILE_DIR = Path("../target/profile")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

COLOR_PALETTE = [
    "#0173B2", "#DE8F05", "#029E73", "#CC78BC",
    "#CA9161", "#949494", "#ECE133", "#56B4E9",
]

TIME_UNITS = [("ms", 1_000_000), ("\u00b5s", 1_000), ("\u03bcs", 1_000), ("ns", 1), ("s", 1_000_000_000)]
BYTES_UNITS = [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]


def parse_time(s: str) -> float | None:
    s = s.strip()
    for suffix, mult in TIME_UNITS:
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)].strip()) * mult
            except ValueError:
                return None
    return None


def parse_bytes_val(s: str) -> float | None:
    s = s.strip()
    for suffix, mult in BYTES_UNITS:
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)].strip()) * mult
            except ValueError:
                return None
    return None


def parse_table(text: str, parse_val) -> dict[str, tuple[float, float | None]]:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("+-") or "Function" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5:
            continue
        fn_name = parts[0]
        val = parse_val(parts[2])
        try:
            pct = float(parts[4].rstrip("%"))
        except ValueError:
            pct = None
        if fn_name and val is not None:
            result[fn_name] = (val, pct)
    return result


def parse_profile_file(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    timing_m = re.search(r"timing - Function execution time metrics\.\n(.*?)(?=\nalloc|\Z)", text, re.DOTALL)
    alloc_m = re.search(r"alloc - Cumulative allocations.*?\n(.*?)(?=\nthreads|\Z)", text, re.DOTALL)
    rss_m = re.search(r"RSS: ([0-9.]+ [A-Z]+)", text)

    timing = parse_table(timing_m.group(1), parse_time) if timing_m else {}
    alloc = parse_table(alloc_m.group(1), parse_bytes_val) if alloc_m else {}
    rss = parse_bytes_val(rss_m.group(1)) if rss_m else None

    if not timing and not alloc:
        return None
    return {"timing": timing, "alloc": alloc, "rss": rss}


def discover_profiles(algo: str) -> dict[tuple[int, int], dict]:
    algo_dir = PROFILE_DIR / algo
    if not algo_dir.exists():
        print(f"Warning: no profile data for {algo}")
        return {}
    data = {}
    pattern = re.compile(r"^(\d+)_ef(\d+)\.txt$")
    for f in sorted(algo_dir.iterdir()):
        m = pattern.match(f.name)
        if not m:
            continue
        nodes, ef = int(m.group(1)), int(m.group(2))
        parsed = parse_profile_file(f)
        if parsed:
            data[(nodes, ef)] = parsed
    return data


def algo_fn(data: dict[tuple[int, int], dict]) -> str | None:
    for entry in data.values():
        for fn in entry["timing"]:
            if fn != "profile::main":
                return fn
    return None


def fmt_time(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns/1_000_000_000:.3f}s"
    if ns >= 1_000_000:
        return f"{ns/1_000_000:.2f}ms"
    if ns >= 1_000:
        return f"{ns/1_000:.1f}\u00b5s"
    return f"{ns:.0f}ns"


def fmt_bytes(b: float) -> str:
    if b >= 1024**3:
        return f"{b/1024**3:.2f}GB"
    if b >= 1024**2:
        return f"{b/1024**2:.2f}MB"
    if b >= 1024:
        return f"{b/1024:.1f}KB"
    return f"{b:.0f}B"


def edge_factors_nodes(data: dict) -> tuple[list[int], list[int]]:
    efs = sorted(set(ef for (_, ef) in data))
    ns = sorted(set(n for (n, _) in data))
    return efs, ns


def plot_time_scaling(all_data: dict[str, dict], out_path: Path):
    n_algos = len(all_data)
    fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 5), squeeze=False)

    for ax_i, (algo, data) in enumerate(all_data.items()):
        ax = axes[0][ax_i]
        fn = algo_fn(data)
        efs, _ = edge_factors_nodes(data)

        for ef_i, ef in enumerate(efs):
            xs = sorted(n for (n, e) in data if e == ef)
            ys = []
            for n in xs:
                t = data[(n, ef)]["timing"].get(fn)
                ys.append(t[0] if t else None)
            xs_clean = [x for x, y in zip(xs, ys) if y is not None]
            ys_clean = [y for y in ys if y is not None]
            if xs_clean:
                ax.plot(xs_clean, ys_clean, "o-", color=COLOR_PALETTE[ef_i % len(COLOR_PALETTE)],
                        label=f"ef={ef}", linewidth=1.5, markersize=4)

        ax.set_title(algo, fontweight="bold")
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Time (ns)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Execution Time Scaling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_memory_scaling(all_data: dict[str, dict], out_path: Path):
    n_algos = len(all_data)
    fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 5), squeeze=False)

    for ax_i, (algo, data) in enumerate(all_data.items()):
        ax = axes[0][ax_i]
        fn = algo_fn(data)
        efs, _ = edge_factors_nodes(data)

        for ef_i, ef in enumerate(efs):
            xs = sorted(n for (n, e) in data if e == ef)
            ys = []
            for n in xs:
                m = data[(n, ef)]["alloc"].get(fn)
                ys.append(m[0] if m else None)
            xs_clean = [x for x, y in zip(xs, ys) if y is not None]
            ys_clean = [y for y in ys if y is not None]
            if xs_clean:
                ax.plot(xs_clean, ys_clean, "o-", color=COLOR_PALETTE[ef_i % len(COLOR_PALETTE)],
                        label=f"ef={ef}", linewidth=1.5, markersize=4)

        ax.set_title(algo, fontweight="bold")
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Allocated bytes")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Memory Allocation Scaling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_algo_fraction(all_data: dict[str, dict], out_path: Path):
    n_algos = len(all_data)
    fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 5), squeeze=False)

    for ax_i, (algo, data) in enumerate(all_data.items()):
        ax = axes[0][ax_i]
        fn = algo_fn(data)
        efs, _ = edge_factors_nodes(data)

        for ef_i, ef in enumerate(efs):
            xs = sorted(n for (n, e) in data if e == ef)
            ys = []
            for n in xs:
                t = data[(n, ef)]["timing"].get(fn)
                ys.append(t[1] if t and t[1] is not None else None)
            xs_clean = [x for x, y in zip(xs, ys) if y is not None]
            ys_clean = [y for y in ys if y is not None]
            if xs_clean:
                ax.plot(xs_clean, ys_clean, "o-", color=COLOR_PALETTE[ef_i % len(COLOR_PALETTE)],
                        label=f"ef={ef}", linewidth=1.5, markersize=4)

        ax.set_title(algo, fontweight="bold")
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Algorithm % of total time")
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Algorithm vs Setup Time (% of profile::main)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_comparison(all_data: dict[str, dict], out_path: Path):
    if len(all_data) < 2:
        return

    algos = list(all_data.keys())
    base = algos[0]
    efs = sorted(set(ef for (_, ef) in all_data[base]))

    fig, axes = plt.subplots(1, len(efs), figsize=(5 * len(efs), 5), squeeze=False)

    for ef_i, ef in enumerate(efs):
        ax = axes[0][ef_i]
        base_fn = algo_fn(all_data[base])

        for ci, comp in enumerate(algos[1:]):
            comp_fn = algo_fn(all_data[comp])
            common_ns = sorted(
                n for n in set(k[0] for k in all_data[base] if k[1] == ef)
                & set(k[0] for k in all_data[comp] if k[1] == ef)
            )
            xs, ys = [], []
            for n in common_ns:
                bt = all_data[base][(n, ef)]["timing"].get(base_fn)
                ct = all_data[comp][(n, ef)]["timing"].get(comp_fn)
                if bt and ct and ct[0] > 0:
                    xs.append(n)
                    ys.append(bt[0] / ct[0])
            if xs:
                ax.plot(xs, ys, "o-", color=COLOR_PALETTE[ci % len(COLOR_PALETTE)],
                        label=f"{base}/{comp}", linewidth=1.5, markersize=4)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"ef={ef}", fontweight="bold")
        ax.set_xlabel("Nodes")
        ax.set_ylabel(f"Time ratio ({base} / other)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Cross-Algorithm Time Ratio", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def generate_report(all_data: dict[str, dict]) -> str:
    lines = [
        "# Profile Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    for algo, data in all_data.items():
        fn = algo_fn(data)
        efs, ns = edge_factors_nodes(data)

        lines.append(f"## {algo}\n")
        if fn:
            lines.append(f"Algorithm function: `{fn}`\n")

        lines.append("### Execution Time\n")
        header = "| nodes |" + "".join(f" ef={ef} | % total |" for ef in efs)
        sep = "| ---: |" + "| ---: | ---: |" * len(efs)
        lines += [header, sep]
        for n in ns:
            row = f"| {n} |"
            for ef in efs:
                t = data.get((n, ef), {}).get("timing", {}).get(fn) if fn else None
                row += f" {fmt_time(t[0])} | {t[1]:.1f}% |" if t else " - | - |"
            lines.append(row)

        lines.append(f"\n![Time scaling](profile_time.png)\n")

        lines.append("### Memory Allocation\n")
        lines += [header.replace("ef=", "ef="), sep]
        for n in ns:
            row = f"| {n} |"
            for ef in efs:
                m = data.get((n, ef), {}).get("alloc", {}).get(fn) if fn else None
                row += f" {fmt_bytes(m[0])} | {m[1]:.1f}% |" if m else " - | - |"
            lines.append(row)

        lines.append(f"\n![Memory scaling](profile_memory.png)\n")

        lines.append("### Bottleneck Summary\n")
        lines.append("**% total** = fraction of `profile::main` time/memory in the algorithm function.")
        lines.append("High % = algorithm dominates. Low % = graph construction dominates.\n")

        pcts = [
            (data[(n, ef)]["timing"][fn][1], n, ef)
            for (n, ef) in data
            if fn and fn in data[(n, ef)]["timing"] and data[(n, ef)]["timing"][fn][1] is not None
        ]
        if pcts:
            pcts.sort()
            lo = pcts[0]
            hi = pcts[-1]
            lines.append(f"- **Lowest:** {lo[0]:.1f}% at nodes={lo[1]}, ef={lo[2]} — graph construction dominates")
            lines.append(f"- **Highest:** {hi[0]:.1f}% at nodes={hi[1]}, ef={hi[2]} — algorithm dominates\n")

        lines.append(f"![Algorithm fraction](profile_algo_fraction.png)\n")

    if len(all_data) >= 2:
        algos = list(all_data.keys())
        lines.append("## Cross-Algorithm Comparison\n")
        lines.append(f"![Comparison](profile_comparison.png)\n")

        common = set(all_data[algos[0]].keys())
        for a in algos[1:]:
            common &= set(all_data[a].keys())

        for ef in sorted(set(ef for (_, ef) in common)):
            ns_ef = sorted(n for (n, e) in common if e == ef)
            lines.append(f"### Edge Factor {ef}\n")
            header = "| nodes |" + "".join(f" {a} time | {a} mem |" for a in algos)
            sep = "| ---: |" + "| ---: | ---: |" * len(algos)
            lines += [header, sep]
            for n in ns_ef:
                row = f"| {n} |"
                for a in algos:
                    fn = algo_fn(all_data[a])
                    t = all_data[a][(n, ef)]["timing"].get(fn) if fn else None
                    m = all_data[a][(n, ef)]["alloc"].get(fn) if fn else None
                    row += f" {fmt_time(t[0]) if t else '-'} | {fmt_bytes(m[0]) if m else '-'} |"
                lines.append(row)
            lines.append("")

    return "\n".join(lines)


def main():
    algos = sys.argv[1:] if len(sys.argv) > 1 else ["dijkstra", "bmssp_base"]

    all_data: dict[str, dict] = {}
    for algo in algos:
        data = discover_profiles(algo)
        if data:
            all_data[algo] = data
            print(f"Loaded {len(data)} profile files for {algo}")
        else:
            print(f"No new-format profile data for {algo} (need <nodes>_ef<ef>.txt files)")

    if not all_data:
        print("No profile data found. Run ./profile.sh <algo> first.")
        sys.exit(1)

    plot_time_scaling(all_data, OUT_DIR / "profile_time.png")
    plot_memory_scaling(all_data, OUT_DIR / "profile_memory.png")
    plot_algo_fraction(all_data, OUT_DIR / "profile_algo_fraction.png")
    if len(all_data) >= 2:
        plot_comparison(all_data, OUT_DIR / "profile_comparison.png")

    report = generate_report(all_data)
    report_path = OUT_DIR / "profile_report.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
