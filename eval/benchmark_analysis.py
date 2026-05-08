# %% [markdown]
# # Benchmark Analysis: Dijkstra vs BMSSP
#
# This notebook visualizes Criterion benchmark results comparing Dijkstra and BMSSP algorithms
# across different graph sizes and edge densities.

# %% Imports
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

# %% Configuration
CRITERION_DIR = Path("../target/criterion")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

COLOR_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#CC78BC",
    "#CA9161",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


# %% Discovery Functions
def discover_algorithms(criterion_dir: Path) -> list[str]:
    if not criterion_dir.exists():
        print(f"Warning: Criterion directory {criterion_dir} does not exist")
        return []

    algorithms = []
    for path in criterion_dir.iterdir():
        if path.is_dir() and path.name.endswith("_comprehensive"):
            algo_name = path.name.replace("_comprehensive", "")
            algorithms.append(algo_name)

    return sorted(algorithms)


def discover_benchmark_configs(
    criterion_dir: Path, algorithm: str
) -> tuple[list[int], list[str]]:
    bench_dir = criterion_dir / f"{algorithm}_comprehensive" / "comprehensive"

    if not bench_dir.exists():
        return [], []

    node_counts = set()
    edge_factors = set()

    for path in bench_dir.iterdir():
        if path.is_dir() and path.name.startswith("size_"):
            parts = path.name.split("_")
            if len(parts) >= 4 and parts[0] == "size" and parts[2] == "edges":
                try:
                    nodes = int(parts[1])
                    edge_factor = parts[3]
                    node_counts.add(nodes)
                    edge_factors.add(edge_factor)
                except ValueError:
                    continue

    return sorted(node_counts), sorted(edge_factors)


def assign_colors(algorithms: list[str]) -> dict[str, str]:
    return {
        algo: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, algo in enumerate(algorithms)
    }


# %% Data Structure
@dataclass
class BenchmarkData:
    mean: float
    ci_lower: float
    ci_upper: float
    tukey_lower: Optional[float] = None
    tukey_upper: Optional[float] = None


# %% Data Loading
def load_benchmark_data(
    algorithm: str, nodes: int, edge_factor: str
) -> Optional[BenchmarkData]:
    bench_name = f"{algorithm}_comprehensive"
    size_name = f"size_{nodes}_edges_{edge_factor}"
    base_path = CRITERION_DIR / bench_name / "comprehensive" / size_name / "new"

    estimates_path = base_path / "estimates.json"
    tukey_path = base_path / "tukey.json"

    if not estimates_path.exists():
        print(f"Warning: Missing {estimates_path}")
        return None

    try:
        with open(estimates_path) as f:
            estimates = json.load(f)

        mean = estimates["mean"]["point_estimate"]
        ci_lower = estimates["mean"]["confidence_interval"]["lower_bound"]
        ci_upper = estimates["mean"]["confidence_interval"]["upper_bound"]

        tukey_lower = None
        tukey_upper = None
        if tukey_path.exists():
            with open(tukey_path) as f:
                tukey = json.load(f)
            tukey_lower = tukey[0]
            tukey_upper = tukey[3]

        return BenchmarkData(
            mean=mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            tukey_lower=tukey_lower,
            tukey_upper=tukey_upper,
        )

    except (json.JSONDecodeError, KeyError) as e:
        print(
            f"Error loading benchmark data for {algorithm} size={nodes} edges={edge_factor}: {e}"
        )
        return None


def load_all_benchmarks(
    algorithms: list[str], node_counts: list[int], edge_factors: list[str]
) -> Dict[str, Dict[str, Dict[int, BenchmarkData]]]:
    data = {}

    for edge_factor in edge_factors:
        data[edge_factor] = {}
        for algorithm in algorithms:
            data[edge_factor][algorithm] = {}
            for nodes in node_counts:
                bench_data = load_benchmark_data(algorithm, nodes, edge_factor)
                if bench_data:
                    data[edge_factor][algorithm][nodes] = bench_data

    return data


# %% Plotting Utilities
def format_time_axis(max_value: float) -> tuple[float, str]:
    if max_value < 1_000:
        return (1.0, "ns")
    elif max_value < 1_000_000:
        return (1_000.0, "µs")
    elif max_value < 1_000_000_000:
        return (1_000_000.0, "ms")
    else:
        return (1_000_000_000.0, "s")


def format_time(ns: float) -> str:
    if ns > 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    elif ns > 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    elif ns > 1_000:
        return f"{ns / 1_000:.3f} us"
    else:
        return f"{ns:.1f} ns"


def is_interactive() -> bool:
    try:
        get_ipython()  # type: ignore # noqa: F821
        return True
    except NameError:
        return False


def should_show_tukey(data: BenchmarkData) -> bool:
    if data.tukey_lower is None or data.tukey_upper is None:
        return False

    ci_range = data.ci_upper - data.ci_lower
    lower_diff = abs(data.tukey_lower - data.ci_lower)
    upper_diff = abs(data.tukey_upper - data.ci_upper)

    return (lower_diff > 0.1 * ci_range) or (upper_diff > 0.1 * ci_range)


# %% Main Plotting Function
def plot_benchmark_comparison(
    data: Dict[str, Dict[int, BenchmarkData]],
    edge_factor: str,
    node_counts: list[int],
    algorithms: list[str],
    colors: dict[str, str],
    output_path: Optional[Path] = None,
):
    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(node_counts))
    width = 0.35 if len(algorithms) == 2 else 0.8 / len(algorithms)

    all_means = []
    for algo_data in data.values():
        all_means.extend([d.mean for d in algo_data.values()])

    if not all_means:
        print(f"Warning: No data available for edge factor {edge_factor}")
        return

    max_mean = max(all_means)
    scale_factor, unit = format_time_axis(max_mean)

    for i, algorithm in enumerate(algorithms):
        if algorithm not in data:
            continue

        algo_data = data[algorithm]
        means = []
        lower_errors = []
        upper_errors = []

        for nodes in node_counts:
            if nodes in algo_data:
                d = algo_data[nodes]
                means.append(d.mean / scale_factor)
                lower_errors.append((d.mean - d.ci_lower) / scale_factor)
                upper_errors.append((d.ci_upper - d.mean) / scale_factor)
            else:
                means.append(0)
                lower_errors.append(0)
                upper_errors.append(0)

        yerr = [lower_errors, upper_errors]

        ax.bar(
            x + i * width,
            means,
            width,
            yerr=yerr,
            label=algorithm.upper(),
            color=colors[algorithm],
            capsize=5,
            alpha=0.8,
            error_kw={"linewidth": 1.5},
        )

        for j, nodes in enumerate(node_counts):
            if nodes in algo_data and should_show_tukey(algo_data[nodes]):
                d = algo_data[nodes]
                bar_x = x[j] + i * width

                ax.hlines(
                    d.tukey_lower / scale_factor,
                    bar_x - width / 2,
                    bar_x + width / 2,
                    colors=colors[algorithm],
                    linestyles="dashed",
                    linewidth=1,
                    alpha=0.5,
                )
                ax.hlines(
                    d.tukey_upper / scale_factor,
                    bar_x - width / 2,
                    bar_x + width / 2,
                    colors=colors[algorithm],
                    linestyles="dashed",
                    linewidth=1,
                    alpha=0.5,
                )

    ax.set_xlabel("Number of Nodes", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Execution Time ({unit})", fontsize=12, fontweight="bold")

    algo_names = " vs ".join(algo.upper() for algo in algorithms)
    ax.set_title(
        f"{algo_names} Performance\nEdge Factor: {edge_factor} (edges ≈ {edge_factor[0]}×nodes)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close()
    else:
        plt.show()


# %% Report Generation
def generate_report(
    all_data: Dict[str, Dict[str, Dict[int, BenchmarkData]]],
    algorithms: list[str],
    node_counts: list[int],
    edge_factors: list[str],
) -> str:
    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- **Algorithms:** {', '.join(a.upper() for a in algorithms)}")
    lines.append(f"- **Node counts:** {', '.join(str(n) for n in node_counts)}")
    lines.append(f"- **Edge factors:** {', '.join(edge_factors)}")
    lines.append("")

    for edge_factor in edge_factors:
        if edge_factor not in all_data:
            continue
        ef_data = all_data[edge_factor]

        lines.append(f"## Edge Factor {edge_factor}")
        lines.append("")

        header = "| Size |"
        separator = "| ---: |"
        for algo in algorithms:
            header += f" {algo.upper()} |"
            separator += " ---: |"
        lines.append(header)
        lines.append(separator)

        baseline_algo = "dijkstra" if "dijkstra" in ef_data else None
        for nodes in node_counts:
            dijkstra_mean = None
            if baseline_algo in ef_data and nodes in ef_data[baseline_algo]:
                dijkstra_mean = ef_data[baseline_algo][nodes].mean
            row = f"| {nodes} |"
            for algo in algorithms:
                if algo in ef_data and nodes in ef_data[algo]:
                    time_str = format_time(ef_data[algo][nodes].mean)
                    if dijkstra_mean is not None and algo != baseline_algo:
                        ratio = ef_data[algo][nodes].mean / dijkstra_mean
                        cell = f"{time_str} ({ratio:.2f}x)"
                    else:
                        cell = time_str
                    row += f" {cell} |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")
        lines.append(f"![](edge_factor_{edge_factor}.png)")
        lines.append("")

    bmssp_algos = [a for a in algorithms if a.startswith("bmssp")]
    if len(bmssp_algos) >= 2:
        lines.append("## Pairwise Speedups")
        lines.append("")

        for i in range(len(bmssp_algos) - 1):
            base = bmssp_algos[i]
            compare = bmssp_algos[i + 1]
            lines.append(f"### {base.upper()} vs {compare.upper()}")
            lines.append("")

            for edge_factor in edge_factors:
                if edge_factor not in all_data:
                    continue
                ef_data = all_data[edge_factor]

                lines.append(f"**Edge Factor {edge_factor}**")
                lines.append("")
                lines.append(f"| Size | {base.upper()} | {compare.upper()} | Speedup | Change |")
                lines.append("| ---: | ---: | ---: | ---: | ---: |")

                for nodes in node_counts:
                    b = ef_data.get(base, {}).get(nodes)
                    c = ef_data.get(compare, {}).get(nodes)
                    if b and c:
                        speedup = b.mean / c.mean
                        pct = (1 - c.mean / b.mean) * 100
                        lines.append(
                            f"| {nodes} | {format_time(b.mean)} | {format_time(c.mean)}"
                            f" | {speedup:.2f}x | {pct:+.1f}% |"
                        )

                lines.append("")

    return "\n".join(lines)


# %% Generate Plots
if __name__ == "__main__":
    print("Discovering benchmarks...")

    algorithms = discover_algorithms(CRITERION_DIR)
    if not algorithms:
        print(f"Error: No benchmark data found in {CRITERION_DIR}")
        print("Run benchmarks first with: cargo bench")
    else:
        print(f"Found algorithms: {', '.join(algorithms)}")

        all_node_counts = set()
        all_edge_factors = set()

        for algo in algorithms:
            nodes, factors = discover_benchmark_configs(CRITERION_DIR, algo)
            all_node_counts.update(nodes)
            all_edge_factors.update(factors)

        node_counts = sorted(all_node_counts)
        edge_factors = sorted(all_edge_factors)

        print(f"Node counts: {node_counts}")
        print(f"Edge factors: {edge_factors}")

        colors = assign_colors(algorithms)

        print("\nLoading benchmark data...")
        all_data = load_all_benchmarks(algorithms, node_counts, edge_factors)

        print(
            f"Running in {'interactive' if is_interactive() else 'non-interactive'} mode"
        )

        for edge_factor in edge_factors:
            if edge_factor not in all_data:
                print(f"Warning: No data for edge factor {edge_factor}")
                continue

            print(f"\nGenerating plot for edge factor {edge_factor}...")

            if is_interactive():
                plot_benchmark_comparison(
                    all_data[edge_factor],
                    edge_factor,
                    node_counts,
                    algorithms,
                    colors,
                )
            else:
                output_path = OUT_DIR / f"edge_factor_{edge_factor}.png"
                plot_benchmark_comparison(
                    all_data[edge_factor],
                    edge_factor,
                    node_counts,
                    algorithms,
                    colors,
                    output_path,
                )

        report = generate_report(all_data, algorithms, node_counts, edge_factors)
        report_path = OUT_DIR / "report.md"
        report_path.write_text(report)
        print(f"\nReport written to {report_path}")

        print("\nDone!")

# %%
