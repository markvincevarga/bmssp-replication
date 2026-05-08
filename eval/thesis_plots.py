from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "msc-thesis" / "figures"

ALGORITHMS = [
    "dijkstra",
    "dijkstra_opt",
    "bmssp_base",
    "bmssp_o_1",
    "bmssp_o_2",
    "bmssp_o_3",
    "bmssp_o_4",
    "bmssp_o_5",
    "bmssp_o_6",
]

ALGO_DISPLAY = {
    "dijkstra": "Dijkstra",
    "dijkstra_opt": "Dijkstra (opt)",
    "bmssp_base": "BMSSP (base)",
    "bmssp_o_1": "BMSSP $o_1$",
    "bmssp_o_2": "BMSSP $o_2$",
    "bmssp_o_3": "BMSSP $o_3$",
    "bmssp_o_4": "BMSSP $o_4$",
    "bmssp_o_5": "BMSSP $o_5$",
    "bmssp_o_6": "BMSSP $o_6$",
}

# KTH brand colors (lib/kthcolors.tex).
KTH = {
    "deep_sea":        "#1954A6",  # primary blue
    "deep_sea_80":     "#5E87C0",
    "stratosphere":    "#2191C4",  # light blue
    "fluorescence":    "#D02F80",  # pink / red
    "fluorescence_80": "#D95599",
    "front_lawn":      "#62922E",  # olive green
    "office":          "#65656C",  # cool gray
    "office_80":       "#848489",
    "black":           "#000000",
}

ALGO_STYLE = {
    "dijkstra":     {"color": KTH["office_80"],       "linestyle": "--",  "linewidth": 2.0, "marker": "s", "zorder": 2, "alpha": 1.0},
    "dijkstra_opt": {"color": KTH["black"],           "linestyle": "--",  "linewidth": 2.5, "marker": "D", "zorder": 5, "alpha": 1.0},
    "bmssp_base":   {"color": KTH["fluorescence"],    "linestyle": "-",   "linewidth": 2.2, "marker": "o", "zorder": 5, "alpha": 1.0},
    "bmssp_o_1":    {"color": KTH["fluorescence_80"], "linestyle": ":",   "linewidth": 1.0, "marker": "^", "zorder": 2, "alpha": 0.55},
    "bmssp_o_2":    {"color": KTH["front_lawn"],      "linestyle": ":",   "linewidth": 1.0, "marker": "v", "zorder": 2, "alpha": 0.55},
    "bmssp_o_3":    {"color": KTH["stratosphere"],    "linestyle": ":",   "linewidth": 1.0, "marker": "<", "zorder": 2, "alpha": 0.55},
    "bmssp_o_4":    {"color": KTH["deep_sea_80"],     "linestyle": ":",   "linewidth": 1.0, "marker": "P", "zorder": 2, "alpha": 0.55},
    "bmssp_o_5":    {"color": KTH["office"],          "linestyle": ":",   "linewidth": 1.0, "marker": ">", "zorder": 2, "alpha": 0.55},
    "bmssp_o_6":    {"color": KTH["deep_sea"],        "linestyle": "-",   "linewidth": 2.2, "marker": "X", "zorder": 5, "alpha": 1.0},
}

EDGE_FACTORS = ["4n", "8n"]
RATIO_BASELINE = "dijkstra_opt"
MEMORY_PLOT_ALGOS = ["dijkstra_opt", "bmssp_base", "bmssp_o_6"]

CROSS_TOPO_SOURCES = [
    ("ef4",       "ER (ef=4)",        "er",   "o",  "-"),
    ("ef8",       "ER (ef=8)",        "er",   "s",  "--"),
    ("ba_m2",     "BA (m=2)",         "ba",   "o",  "-"),
    ("ba_m4",     "BA (m=4)",         "ba",   "s",  "--"),
    ("ws_k4_b30", "WS (k=4, \u03b2=.3)",  "ws",   "^",  "--"),
    ("ws_k8_b30", "WS (k=8, \u03b2=.3)",  "ws",   "s",  ":"),
    ("grid",      "Grid",             "grid", "o",  "-"),
]

FAMILY_COLORS = {
    "er":   KTH["deep_sea"],     # primary blue
    "ba":   KTH["fluorescence"], # pink
    "ws":   KTH["front_lawn"],   # olive
    "rg":   KTH["stratosphere"], # light blue
    "grid": KTH["office"],       # cool gray
}

TOPOLOGY_GROUPS = [
    {
        "key": "fixed",
        "title": "Erd\u0151s\u2013R\u00e9nyi (ER)",
        "sources": [("ef4", "ef=4"), ("ef8", "ef=8")],
    },
    {
        "key": "barabasi_albert",
        "title": "Barab\u00e1si\u2013Albert",
        "sources": [("ba_m2", "m=2"), ("ba_m4", "m=4")],
    },
    {
        "key": "watts_strogatz",
        "title": "Watts\u2013Strogatz",
        "sources": [("ws_k4_b30", "k=4, \u03b2=0.3"), ("ws_k8_b30", "k=8, \u03b2=0.3")],
    },
    {
        "key": "spatial",
        "title": "Spatial & Structured",
        "sources": [("rg_r50", "RG (r=0.05)"), ("rg_r6", "RG (r=0.006)"), ("grid", "Grid")],
    },
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (5.5, 3.8),
    "figure.dpi": 300,
    "text.usetex": False,
})



class Data:
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self._results: dict = raw["results"]
        hw = raw.get("hardware", {})
        cpu = hw.get("cpu", "unknown")
        arch = hw.get("arch", "")
        self.hw_label: str = f"{cpu} ({arch})" if arch else cpu
        self.hw_label_tex: str = self.hw_label.replace("_", r"\_")
        self.hw_short: str = cpu.split()[0] if cpu != "unknown" else "unknown"

    def _dig(self, algo: str, n: int, source_key: str, *keys: str):
        try:
            node = self._results[algo][source_key][str(n)]
            for k in keys:
                node = node[k]
            return node
        except (KeyError, TypeError):
            return None

    def mean(self, algo: str, n: int, source_key: str) -> float | None:
        return self._dig(algo, n, source_key, "runtime_ns", "mean", "point_estimate")

    def mean_stderr(self, algo: str, n: int, source_key: str) -> float | None:
        return self._dig(algo, n, source_key, "runtime_ns", "mean", "standard_error")

    def memory(self, algo: str, n: int, source_key: str) -> int | None:
        return self._dig(algo, n, source_key, "memory_bytes")

    def memory_stddev(self, algo: str, n: int, source_key: str) -> float | None:
        return self._dig(algo, n, source_key, "memory_stats", "stddev")

    def sizes(self, algo: str, source_key: str) -> list[int]:
        try:
            return sorted(int(s) for s in self._results[algo][source_key])
        except (KeyError, TypeError):
            return []

    def common_sizes(self, algos: list[str], source_key: str) -> list[int]:
        sets = [s for a in algos if (s := set(self.sizes(a, source_key)))]
        return sorted(set.intersection(*sets)) if sets else []

    def ratio(self, algo: str, baseline: str, n: int, source_key: str) -> float | None:
        v = self.mean(algo, n, source_key)
        b = self.mean(baseline, n, source_key)
        if v is not None and b is not None and b > 0:
            return v / b
        return None

    def ratio_log_se(self, algo: str, baseline: str, n: int, source_key: str) -> float | None:
        v = self.mean(algo, n, source_key)
        b = self.mean(baseline, n, source_key)
        se_v = self.mean_stderr(algo, n, source_key)
        se_b = self.mean_stderr(baseline, n, source_key)
        if v is None or b is None or se_v is None or se_b is None or v <= 0 or b <= 0:
            return None
        return math.sqrt((se_v / v) ** 2 + (se_b / b) ** 2)

    def memory_ratio(self, algo: str, baseline: str, n: int, source_key: str) -> float | None:
        v = self.memory(algo, n, source_key)
        b = self.memory(baseline, n, source_key)
        if v is not None and b is not None and b > 0:
            return v / b
        return None


class MultiData:
    def __init__(self, datasets: list[Data]):
        self.datasets = datasets

    @property
    def hw_labels(self) -> list[str]:
        return [d.hw_label for d in self.datasets]

    def common_sizes(self, algos: list[str], source_key: str) -> list[int]:
        sets = [set(d.common_sizes(algos, source_key)) for d in self.datasets]
        if len(sets) != len(self.datasets) or not all(sets):
            return []
        return sorted(set.intersection(*sets))

    def _geomean(self, values: list[float]) -> float:
        product = 1.0
        for v in values:
            product *= v
        return product ** (1.0 / len(values))

    def merged_ratio(self, algo: str, baseline: str, n: int, source_key: str) -> float | None:
        ratios = [d.ratio(algo, baseline, n, source_key) for d in self.datasets]
        if any(r is None for r in ratios):
            return None
        return self._geomean(ratios)

    def merged_ratio_ci(self, algo: str, baseline: str, n: int, source_key: str,
                        z: float = 1.96) -> tuple[float, float] | None:
        g = self.merged_ratio(algo, baseline, n, source_key)
        log_ses = [d.ratio_log_se(algo, baseline, n, source_key) for d in self.datasets]
        if g is None or any(s is None for s in log_ses):
            return None
        k = len(log_ses)
        log_se_g = math.sqrt(sum(s * s for s in log_ses)) / k
        half = z * log_se_g
        return g * math.exp(-half), g * math.exp(half)

    def merged_memory_ratio(self, algo: str, baseline: str, n: int, source_key: str) -> float | None:
        ratios = [d.memory_ratio(algo, baseline, n, source_key) for d in self.datasets]
        if any(r is None for r in ratios):
            return None
        return self._geomean(ratios)


EF_LABEL_TO_SOURCE = {"4n": "ef4", "8n": "ef8"}

OPT_ALGOS = ["bmssp_base", "bmssp_o_1", "bmssp_o_2", "bmssp_o_3",
             "bmssp_o_4", "bmssp_o_5", "bmssp_o_6"]


def _filter_none(sizes, vals):
    pairs = [(s, v) for s, v in zip(sizes, vals) if v is not None]
    return zip(*pairs) if pairs else ([], [])


def _collect_ratio(ratio_fn, algo: str, baseline: str, sizes, source_key: str):
    pts = [(n, r) for n in sizes if (r := ratio_fn(algo, baseline, n, source_key)) is not None]
    return zip(*pts) if pts else ([], [])


def _save(fig, out: Path, **kwargs):
    fig.savefig(out, **kwargs)
    plt.close(fig)
    print(f"  Saved {out}")


def _write_tex(path: Path, lines: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}")


def _latex_table(caption: str, label: str, col_spec: str, header: str, body: list[str], *,
                 resizebox: bool = False) -> list[str]:
    pre = [r"  \resizebox{\textwidth}{!}{%"] if resizebox else []
    post = [r"  }"] if resizebox else []
    return [
        r"\begin{table}[!ht]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        *pre,
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}" + ("%" if resizebox else ""),
        *post,
        r"\end{table}",
    ]


def format_nodes(n: int) -> str:
    if n > 0 and (n & (n - 1)) == 0:
        exp = int(math.log2(n))
        return f"$2^{{{exp}}}$"
    return str(n)


def ns_to_label(ns: float) -> tuple[float, str]:
    if ns >= 1e9:
        return 1e9, "s"
    if ns >= 1e6:
        return 1e6, "ms"
    if ns >= 1e3:
        return 1e3, "\u00b5s"
    return 1.0, "ns"


def _setup_xaxis(ax, sizes):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: format_nodes(int(v)) if v == int(v) else "")
    ax.xaxis.set_major_formatter(fmt)
    if sizes:
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(sizes))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def setup_axes(ax, sizes, all_vals):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vertices $n$")

    scale, unit = ns_to_label(max(all_vals)) if all_vals else (1e6, "ms")
    ax.set_ylabel(f"Execution time ({unit})")

    _setup_xaxis(ax, sizes)

    yfmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/scale:.3g}")
    ax.yaxis.set_major_formatter(yfmt)
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    return scale, unit


def setup_ratio_axes(ax, sizes):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vertices $n$")
    ax.set_ylabel(f"Time ratio vs {ALGO_DISPLAY[RATIO_BASELINE]}")

    _setup_xaxis(ax, sizes)
    ax.axhline(y=1.0, color=KTH["black"], linewidth=0.8, linestyle=":", alpha=0.6)
    # Explicit log ticks so 1x, 2x, 5x, 10x, 20x read directly.
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.5, 2, 3, 5, 7, 10, 15, 20]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, which="major", alpha=0.3, linestyle="--")



def plot_absolute(data: Data, ef: str, out_dir: Path, *, suffix: str = ""):
    source_key = EF_LABEL_TO_SOURCE.get(ef)
    if not source_key:
        return
    core = ["dijkstra", "dijkstra_opt", "bmssp_base", "bmssp_o_6"]
    sizes = data.common_sizes(core, source_key)
    if not sizes:
        print(f"  No common sizes for core algorithms at {ef}")
        return

    fig, ax = plt.subplots()
    all_vals = []
    for algo in ALGORITHMS:
        vals = [data.mean(algo, n, source_key) for n in sizes]
        xs, ys = _filter_none(sizes, vals)
        if ys:
            all_vals.extend(ys)
            ax.plot(xs, ys, label=ALGO_DISPLAY[algo], markersize=5, **ALGO_STYLE[algo])

    setup_axes(ax, sizes, all_vals)
    if suffix:
        ax.set_title(data.hw_label, fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / f"time_abs_{ef}{suffix}.pdf")


def _plot_ratio_on_ax(ax, ratio_fn, source_key: str, sizes):
    for algo in ALGORITHMS:
        xs, ys = _collect_ratio(ratio_fn, algo, RATIO_BASELINE, sizes, source_key)
        if ys:
            ax.plot(xs, ys, label=ALGO_DISPLAY[algo], markersize=5, **ALGO_STYLE[algo])
    setup_ratio_axes(ax, sizes)


def plot_ratio(data: Data, ef: str, out_dir: Path, *, suffix: str = ""):
    source_key = EF_LABEL_TO_SOURCE.get(ef)
    if not source_key:
        return
    core = [RATIO_BASELINE, "bmssp_base", "bmssp_o_6"]
    sizes = data.common_sizes(core, source_key)
    if not sizes:
        print(f"  No common sizes for ratio at {ef}")
        return

    fig, ax = plt.subplots()
    _plot_ratio_on_ax(ax, data.ratio, source_key, sizes)
    if suffix:
        ax.set_title(data.hw_label, fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / f"time_ratio_{ef}{suffix}.pdf")


def plot_ratio_merged(multi: MultiData, ef: str, out_dir: Path):
    source_key = EF_LABEL_TO_SOURCE.get(ef)
    if not source_key:
        return
    core = [RATIO_BASELINE, "bmssp_base", "bmssp_o_6"]
    sizes = multi.common_sizes(core, source_key)
    if not sizes:
        print(f"  No common sizes for merged ratio at {ef}")
        return

    fig, ax = plt.subplots()
    _plot_ratio_on_ax(ax, multi.merged_ratio, source_key, sizes)
    ax.set_title("Geometric mean across hardware", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir / f"time_ratio_{ef}_merged.pdf")



def _scaled_style(algo):
    st = ALGO_STYLE[algo]
    return {**st, "linewidth": st["linewidth"] * 0.85}


def _draw_topology_abs_row(ax, data: Data, source_key: str, source_label: str, is_last: bool):
    core_abs = ["dijkstra", "dijkstra_opt", "bmssp_base", "bmssp_o_6"]
    src_sizes = data.common_sizes(core_abs, source_key)
    all_vals = []
    for algo in ALGORITHMS:
        vals = [data.mean(algo, n, source_key) for n in src_sizes]
        xs, ys = _filter_none(src_sizes, vals)
        if ys:
            all_vals.extend(ys)
            ax.plot(xs, ys, markersize=4, **_scaled_style(algo))

    if all_vals:
        ax.set_xscale("log")
        ax.set_yscale("log")
        scale, unit = ns_to_label(max(all_vals))
        yfmt = matplotlib.ticker.FuncFormatter(lambda v, _s=scale: f"{v/_s:.3g}" if _s else "")
        ax.yaxis.set_major_formatter(yfmt)
        ax.set_ylabel(f"Time ({unit})", fontsize=8)

    ax.set_title(f"{source_label} \u2014 absolute", fontsize=9)
    _setup_xaxis(ax, src_sizes)
    ax.grid(True, which="major", alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=7)
    if is_last:
        ax.set_xlabel("$n$", fontsize=8)


def _draw_topology_ratio_row(ax, ratio_fn, common_sizes_fn, source_key: str,
                             source_label: str, is_last: bool, *, title_extra: str = ""):
    core_ratio = [RATIO_BASELINE, "bmssp_base", "bmssp_o_6"]
    ratio_sizes = common_sizes_fn(core_ratio, source_key)
    for algo in ALGORITHMS:
        xs, ys = _collect_ratio(ratio_fn, algo, RATIO_BASELINE, ratio_sizes, source_key)
        if ys:
            ax.plot(xs, ys, markersize=4, **_scaled_style(algo))

    ax.set_xscale("log")
    title_suffix = f" ({title_extra})" if title_extra else ""
    ax.set_title(f"{source_label} \u2014 ratio{title_suffix}", fontsize=9)
    ax.axhline(y=1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    _setup_xaxis(ax, ratio_sizes)
    ax.grid(True, which="major", alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=7)
    ax.set_ylabel(f"Ratio vs {ALGO_DISPLAY[RATIO_BASELINE]}", fontsize=8)
    if is_last:
        ax.set_xlabel("$n$", fontsize=8)


def _add_topology_legend(fig, axes):
    handles = []
    labels = []
    for algo in ALGORITHMS:
        h, = axes[0][0].plot([], [], label=ALGO_DISPLAY[algo], markersize=4, **_scaled_style(algo))
        handles.append(h)
        labels.append(ALGO_DISPLAY[algo])
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))


def plot_topology_group(data: Data, group: dict, out_dir: Path, *, suffix: str = ""):
    sources = group["sources"]
    title = group["title"]
    key = group["key"]

    nrows = len(sources)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.2 * nrows), squeeze=False)

    for row, (source_key, source_label) in enumerate(sources):
        is_last = row == nrows - 1
        _draw_topology_abs_row(axes[row][0], data, source_key, source_label, is_last)
        _draw_topology_ratio_row(axes[row][1], data.ratio, data.common_sizes,
                                 source_key, source_label, is_last)

    _add_topology_legend(fig, axes)
    suptitle = f"{title} \u2014 {data.hw_label}" if suffix else title
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    _save(fig, out_dir / f"topology_{key}{suffix}.pdf", bbox_inches="tight")


def plot_topology_group_merged(multi: MultiData, group: dict, out_dir: Path):
    sources = group["sources"]
    title = group["title"]
    key = group["key"]

    nrows = len(sources)
    ncols = 1 + len(multi.datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)

    for row, (source_key, source_label) in enumerate(sources):
        is_last = row == nrows - 1
        _draw_topology_ratio_row(axes[row][0], multi.merged_ratio, multi.common_sizes,
                                 source_key, source_label, is_last, title_extra="merged")
        for col, d in enumerate(multi.datasets, start=1):
            ax = axes[row][col]
            _draw_topology_ratio_row(ax, d.ratio, d.common_sizes,
                                     source_key, source_label, is_last)
            if row == 0:
                ax.set_title(f"{source_label} \u2014 {d.hw_short}", fontsize=9)

    _add_topology_legend(fig, axes)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    _save(fig, out_dir / f"topology_{key}_merged.pdf", bbox_inches="tight")



def plot_cross_topology_comparison(data: Data, out_dir: Path, *, suffix: str = ""):
    representative_sources = {
        "ef8":       ("ER ($m/n{=}8$)",      "er"),
        "ba_m4":     ("BA ($m{=}4$)",         "ba"),
        "ws_k8_b30": ("WS ($k{=}8,\\beta{=}0.3$)", "ws"),
        "grid":      ("Grid",                 "grid"),
    }

    core_algos = ["dijkstra_opt", "bmssp_o_6"]
    all_sizes = set()
    for sk in representative_sources:
        all_sizes |= set(data.common_sizes(core_algos, sk))
    sizes = sorted(all_sizes)
    if not sizes:
        print("  No data for cross-topology comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False, sharey=True)
    # Walk both panels once to collect the union of values, then run setup_axes
    # against the same range so the shared log-y is consistent across panels.
    panel_vals: list[list[float]] = [[], []]
    panel_lines: list[list[tuple]] = [[], []]
    for col, algo in enumerate(core_algos):
        for source_key, (label, family) in representative_sources.items():
            src_sizes = data.sizes(algo, source_key)
            vals = [data.mean(algo, n, source_key) for n in src_sizes]
            xs, ys = _filter_none(src_sizes, vals)
            if ys:
                panel_vals[col].extend(ys)
                panel_lines[col].append((xs, ys, label, FAMILY_COLORS[family]))

    combined = panel_vals[0] + panel_vals[1]
    for col, algo in enumerate(core_algos):
        ax = axes[0][col]
        for xs, ys, label, color in panel_lines[col]:
            ax.plot(xs, ys, label=label, markersize=5, linewidth=1.8, marker="o", color=color)
        if combined:
            setup_axes(ax, sizes, combined)
        ax.set_title(ALGO_DISPLAY[algo])
        ax.legend(loc="upper left", framealpha=0.9, fontsize=7)

    if suffix:
        fig.suptitle(data.hw_label, fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / f"cross_topology{suffix}.pdf")


def _collect_cross_topo_sizes(data: Data, algos: list[str]) -> list[int]:
    all_sizes: set[int] = set()
    for source_key, *_ in CROSS_TOPO_SOURCES:
        all_sizes |= set(data.common_sizes(algos, source_key))
    return sorted(all_sizes)


def _draw_cross_topo_ratio(ax, ratio_fn, common_sizes_fn, sizes):
    pair = [RATIO_BASELINE, "bmssp_o_6"]
    for source_key, label, family, marker, ls in CROSS_TOPO_SOURCES:
        src_sizes = common_sizes_fn(pair, source_key)
        xs, ys = _collect_ratio(ratio_fn, "bmssp_o_6", RATIO_BASELINE, src_sizes, source_key)
        if ys:
            ax.plot(xs, ys, label=label, markersize=4, linewidth=1.5,
                    color=FAMILY_COLORS[family], marker=marker, linestyle=ls)
    setup_ratio_axes(ax, sizes)


def plot_cross_topology_ratio(data: Data, out_dir: Path, *, suffix: str = ""):
    pair = [RATIO_BASELINE, "bmssp_o_6"]
    sizes = _collect_cross_topo_sizes(data, pair)
    if not sizes:
        print("  No data for cross-topology ratio")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _draw_cross_topo_ratio(ax, data.ratio, data.common_sizes, sizes)
    title = f"{ALGO_DISPLAY['bmssp_o_6']} vs {ALGO_DISPLAY[RATIO_BASELINE]} across topologies"
    if suffix:
        title += f" \u2014 {data.hw_label}"
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / f"cross_topology_ratio{suffix}.pdf")


def plot_cross_topology_ratio_multi(multi: MultiData, out_dir: Path):
    pair = [RATIO_BASELINE, "bmssp_o_6"]
    all_sizes: set[int] = set()
    for source_key, *_ in CROSS_TOPO_SOURCES:
        all_sizes |= set(multi.common_sizes(pair, source_key))
    sizes = sorted(all_sizes)
    if not sizes:
        print("  No data for multi cross-topology ratio")
        return

    ncols = 1 + len(multi.datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.4), squeeze=False, sharey=True)

    _draw_cross_topo_ratio(axes[0][0], multi.merged_ratio, multi.common_sizes, sizes)
    axes[0][0].set_title("Merged (geometric mean)")

    for col, d in enumerate(multi.datasets, start=1):
        per_sizes = _collect_cross_topo_sizes(d, pair)
        _draw_cross_topo_ratio(axes[0][col], d.ratio, d.common_sizes, per_sizes)
        axes[0][col].set_title(d.hw_short)
        axes[0][col].set_ylabel("")

    y_max = max(ax.get_ylim()[1] for ax in axes[0])
    y_min = min(ax.get_ylim()[0] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(y_min, y_max)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{ALGO_DISPLAY['bmssp_o_6']} vs {ALGO_DISPLAY[RATIO_BASELINE]} across topologies", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _save(fig, out_dir / "cross_topology_ratio_merged.pdf", bbox_inches="tight")



def _plot_memory_line(ax, data, algo, sizes, source_key, *, linestyle="-"):
    style = ALGO_STYLE[algo]
    vals = [data.memory(algo, n, source_key) for n in sizes]
    stds = [data.memory_stddev(algo, n, source_key) for n in sizes]
    xs = [s for s, v in zip(sizes, vals) if v is not None]
    ys = [v / 1048576 for v in vals if v is not None]
    sd = [(s or 0) / 1048576 for s, v in zip(stds, vals) if v is not None]
    if not ys:
        return []
    ax.plot(xs, ys, label=ALGO_DISPLAY[algo], markersize=5, linewidth=1.8,
            marker=style["marker"], color=style["color"], linestyle=linestyle)
    if any(s > 0 for s in sd):
        ys_lo = [max(y - s, 0) for y, s in zip(ys, sd)]
        ys_hi = [y + s for y, s in zip(ys, sd)]
        ax.fill_between(xs, ys_lo, ys_hi, alpha=0.15, color=style["color"])
    return ys


def plot_memory_scaling(data: Data, out_dir: Path, *, suffix: str = ""):
    sources = [("ef4", "ef=4"), ("ef8", "ef=8")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for col, (source_key, source_label) in enumerate(sources):
        ax = axes[0][col]
        sizes = data.common_sizes(MEMORY_PLOT_ALGOS, source_key)
        all_vals = []
        for algo in MEMORY_PLOT_ALGOS:
            ys = _plot_memory_line(ax, data, algo, sizes, source_key)
            all_vals.extend(ys)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Vertices $n$")
        ax.set_ylabel("Peak memory (MB)")
        ax.set_title(f"Memory scaling ({source_label})")
        _setup_xaxis(ax, sizes)
        ax.grid(True, which="major", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    if suffix:
        fig.suptitle(data.hw_label, fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / f"memory_scaling{suffix}.pdf")


def _draw_memory_ratio(ax, ratio_fn, common_sizes_fn, sizes):
    pair = ["dijkstra_opt", "bmssp_o_6"]
    for source_key, label, family, marker, ls in CROSS_TOPO_SOURCES:
        src_sizes = common_sizes_fn(pair, source_key)
        xs, ys = _collect_ratio(ratio_fn, "bmssp_o_6", "dijkstra_opt", src_sizes, source_key)
        if ys:
            ax.plot(xs, ys, label=label, markersize=4, linewidth=1.5,
                    color=FAMILY_COLORS[family], marker=marker, linestyle=ls)
    ax.set_xscale("log")
    ax.set_xlabel("Vertices $n$")
    ax.set_ylabel("Memory ratio (BMSSP $o_6$ / Dijkstra opt)")
    ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    _setup_xaxis(ax, sizes)
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    ax.set_ylim(0.95, None)


def plot_memory_ratio(data: Data, out_dir: Path, *, suffix: str = ""):
    pair = ["dijkstra_opt", "bmssp_o_6"]
    sizes = _collect_cross_topo_sizes(data, pair)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _draw_memory_ratio(ax, data.memory_ratio, data.common_sizes, sizes)
    title = "Peak memory ratio across topologies"
    if suffix:
        title += f" \u2014 {data.hw_label}"
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / f"memory_ratio{suffix}.pdf")


def plot_memory_ratio_multi(multi: MultiData, out_dir: Path):
    pair = ["dijkstra_opt", "bmssp_o_6"]
    all_sizes: set[int] = set()
    for source_key, *_ in CROSS_TOPO_SOURCES:
        all_sizes |= set(multi.common_sizes(pair, source_key))
    sizes = sorted(all_sizes)
    if not sizes:
        print("  No data for multi memory ratio")
        return

    ncols = 1 + len(multi.datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.4), squeeze=False, sharey=True)

    _draw_memory_ratio(axes[0][0], multi.merged_memory_ratio, multi.common_sizes, sizes)
    axes[0][0].set_title("Merged (geometric mean)")

    for col, d in enumerate(multi.datasets, start=1):
        per_sizes = _collect_cross_topo_sizes(d, pair)
        _draw_memory_ratio(axes[0][col], d.memory_ratio, d.common_sizes, per_sizes)
        axes[0][col].set_title(d.hw_short)
        axes[0][col].set_ylabel("")

    y_max = max(ax.get_ylim()[1] for ax in axes[0])
    y_min = min(ax.get_ylim()[0] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(y_min, y_max)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Peak memory ratio across topologies", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _save(fig, out_dir / "memory_ratio_merged.pdf", bbox_inches="tight")


def generate_memory_table(data: Data, out_dir: Path, *, suffix: str = ""):
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
    source_key = "ef8"

    body = []
    for n in sizes:
        d = data.memory("dijkstra_opt", n, source_key)
        b = data.memory("bmssp_o_6", n, source_key)
        if d is not None and b is not None:
            n_fmt = f"$2^{{{int(math.log2(n))}}}$"
            body.append(f"    {n_fmt} & {d / 1048576:.1f} & {b / 1048576:.1f} & {b / d:.2f}$\\times$ \\\\")

    label_suffix = f":{suffix.lstrip('_')}" if suffix else ""
    lines = _latex_table(
        f"Peak heap memory of optimized Dijkstra and BMSSP $o_6$ on Erd\\H{{o}}s--R\\'enyi graphs at $m/n = 8$ ($m$ = edges, $n$ = vertices) on {data.hw_label_tex}. Values are mebibytes (MiB); ratio is BMSSP $o_6$ over optimized Dijkstra." if suffix
        else r"Peak heap memory: optimized Dijkstra vs.\ BMSSP $o_6$ on Erd\H{o}s--R\'enyi graphs at $m/n = 8$. Values are mebibytes (MiB); ratio is BMSSP $o_6$ over optimized Dijkstra.",
        f"tab:memory{label_suffix}", "r r r c",
        r"{$n$} & {Dijkstra (MiB)} & {BMSSP $o_6$ (MiB)} & {Ratio}",
        body,
    )
    _write_tex(out_dir / f"memory_table{suffix}.tex", lines)



def generate_summary_table(data: Data, out_dir: Path, *, suffix: str = ""):
    size = 1048576
    source_key = "ef8"
    baseline_algo = "dijkstra_opt"

    baseline_time = data.mean(baseline_algo, size, source_key)
    if baseline_time is None:
        print("  No baseline data for summary table")
        return

    baseline_mem = data.memory(baseline_algo, size, source_key)

    body = []
    for algo in ALGORITHMS:
        t = data.mean(algo, size, source_key)
        m = data.memory(algo, size, source_key)
        if t is None:
            continue
        t_ms = t / 1e6
        ratio_str = f"{t / baseline_time:.2f}$\\times$" if algo != baseline_algo else "1.00$\\times$"
        if m is not None and baseline_mem is not None:
            m_str = f"{m / 1048576:.1f}"
            mr_str = f"{m / baseline_mem:.2f}$\\times$"
        else:
            m_str = "--"
            mr_str = "--"
        body.append(f"    {ALGO_DISPLAY[algo]} & {t_ms:.1f} & {ratio_str} & {m_str} & {mr_str} \\\\")

    label_suffix = f":{suffix.lstrip('_')}" if suffix else ""
    lines = _latex_table(
        f"Per-algorithm runtime and peak memory at $n = 2^{{20}}$ on Erd\\H{{o}}s--R\\'enyi graphs at $m/n = 8$ ($m$ = edges, $n$ = vertices) on {data.hw_label_tex}. Time ratio and memory ratio are relative to optimized Dijkstra. Mem in mebibytes (MiB)." if suffix
        else r"Per-algorithm runtime and peak memory at $n = 2^{20}$ vertices on Erd\H{o}s--R\'enyi graphs at $m/n = 8$ (${\sim}8$ edges per vertex). Time ratio and memory ratio are relative to optimized Dijkstra. Mem in mebibytes (MiB).",
        f"tab:summary{label_suffix}", "l r c r c",
        r"{Algorithm} & {Time (ms)} & {Ratio} & {Peak Mem (MiB)} & {Mem Ratio}",
        body,
    )
    _write_tex(out_dir / f"summary_table{suffix}.tex", lines)


def _fmt_ratio_ci(lo: float, hi: float) -> str:
    """3-decimal precision when both bounds round to 1.00 at 2 decimals;
    otherwise 2 decimals. Avoids the [1.00, 1.00] artifact reviewers
    flagged on the unoptimized-Dijkstra row."""
    if abs(round(lo, 2) - 1.00) < 0.005 and abs(round(hi, 2) - 1.00) < 0.005:
        return f"[{lo:.3f}, {hi:.3f}]"
    return f"[{lo:.2f}, {hi:.2f}]"


def generate_summary_table_merged(multi: MultiData, out_dir: Path):
    size = 1048576
    source_key = "ef8"
    baseline_algo = "dijkstra_opt"

    # siunitx S columns for runtime ratios so decimals align; CI stays as a
    # plain c column because it carries [lo, hi] formatting. Mem ratio
    # column dropped: peak memory has its own table (tab:memory:hw0/hw1)
    # and figure (fig:memory_ratio); leaving Mem ratio here invited
    # readers to mistake near-constant values for runtime constancy.
    n_hw = len(multi.datasets)
    s_cols = " ".join(["S[table-format=1.2]"] * n_hw)
    col_spec = f"l {s_cols} S[table-format=1.2] c"
    hw_headers = " & ".join(f"{{R ({d.hw_short})}}" for d in multi.datasets)
    header = r"{Algorithm} & " + hw_headers + r" & {Merged ratio} & {95\% CI}"

    body = []
    for algo in ALGORITHMS:
        cols = []
        for d in multi.datasets:
            r = d.ratio(algo, baseline_algo, size, source_key)
            cols.append(f"{r:.2f}" if r is not None else "{--}")
        mr = multi.merged_ratio(algo, baseline_algo, size, source_key)
        merged_str = f"{mr:.2f}" if mr is not None else "{--}"
        ci = multi.merged_ratio_ci(algo, baseline_algo, size, source_key)
        ci_str = _fmt_ratio_ci(ci[0], ci[1]) if ci is not None else "--"
        if algo == baseline_algo:
            merged_str = "1.00"
            ci_str = "--"
            cols = ["1.00"] * len(multi.datasets)
        row = " & ".join(cols)
        body.append(f"    {ALGO_DISPLAY[algo]} & {row} & {merged_str} & {ci_str} \\\\")

    lines = _latex_table(
        r"Per-algorithm runtime ratios at $n = 2^{20}$ on Erd\H{o}s--R\'enyi graphs at $m/n = 8$. Time ratios are relative to optimized Dijkstra; merged is the geometric mean of per-platform ratios. 95\% CI propagated from per-platform Criterion bootstrap intervals via the delta method on log-ratios. Peak memory comparison is in Table~\ref{tab:memory} and Figure~\ref{fig:memory_ratio}.",
        "tab:summary:merged", col_spec, header, body,
    )
    _write_tex(out_dir / "summary_table_merged.tex", lines)


TOPOLOGY_SUMMARY_SOURCES = [
    ("ef4",       r"ER ($m/n = 4$)"),
    ("ef8",       r"ER ($m/n = 8$)"),
    ("ba_m2",     r"BA ($m = 2$)"),
    ("ba_m4",     r"BA ($m = 4$)"),
    ("ws_k4_b30", r"WS ($k{=}4, \beta{=}0.3$)"),
    ("ws_k8_b30", r"WS ($k{=}8, \beta{=}0.3$)"),
    ("grid",      r"Grid (4-connected)"),
]


def generate_topology_summary_table(data: Data, out_dir: Path, *, suffix: str = ""):
    size = 1048576
    target_algo = "bmssp_o_6"
    baseline_algo = "dijkstra_opt"

    body = []
    for src_key, label in TOPOLOGY_SUMMARY_SOURCES:
        r = data.ratio(target_algo, baseline_algo, size, src_key)
        ratio_str = f"{r:.2f}$\\times$" if r is not None else "--"
        body.append(f"    {label} & {ratio_str} \\\\")

    label_suffix = f":{suffix.lstrip('_')}" if suffix else ""
    lines = _latex_table(
        f"Per-topology BMSSP $o_6$ / Dijkstra (opt) ratio at $n = 2^{{20}}$ ({data.hw_label_tex})." if suffix
        else r"Per-topology BMSSP $o_6$ / Dijkstra (opt) ratio at $n = 2^{20}$.",
        f"tab:topo:summary{label_suffix}", "l c",
        r"{Topology} & {Ratio}",
        body,
    )
    _write_tex(out_dir / f"topology_summary_table{suffix}.tex", lines)


def generate_topology_summary_table_merged(multi: MultiData, out_dir: Path):
    size = 1048576
    target_algo = "bmssp_o_6"
    baseline_algo = "dijkstra_opt"

    hw_headers = " & ".join(f"{{R ({d.hw_short})}}" for d in multi.datasets)
    col_spec = "l " + " c" * len(multi.datasets) + " c c"
    header = r"{Topology} & " + hw_headers + r" & {Merged} & {95\% CI}"

    body = []
    for src_key, label in TOPOLOGY_SUMMARY_SOURCES:
        cols = []
        for d in multi.datasets:
            r = d.ratio(target_algo, baseline_algo, size, src_key)
            cols.append(f"{r:.2f}$\\times$" if r is not None else "--")
        mr = multi.merged_ratio(target_algo, baseline_algo, size, src_key)
        merged_str = f"{mr:.2f}$\\times$" if mr is not None else "--"
        ci = multi.merged_ratio_ci(target_algo, baseline_algo, size, src_key)
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci is not None else "--"
        row = " & ".join(cols)
        body.append(f"    {label} & {row} & {merged_str} & {ci_str} \\\\")

    lines = _latex_table(
        r"Per-topology BMSSP $o_6$ / Dijkstra (opt) ratio at $n = 2^{20}$. "
        r"Merged is the geometric mean across hardware; 95\% CI is propagated "
        r"from per-platform Criterion bootstrap intervals via the delta method on log-ratios. "
        r"Random geometric graphs are excluded because no single radius maintained "
        r"consistent density across all sizes.",
        "tab:topo:summary:merged", col_spec, header, body,
    )
    _write_tex(out_dir / "topology_summary_table_merged.tex", lines)



_OPT_SHORT = {
    "bmssp_base": "Base",
    "bmssp_o_1": "$o_1$", "bmssp_o_2": "$o_2$", "bmssp_o_3": "$o_3$",
    "bmssp_o_4": "$o_4$", "bmssp_o_5": "$o_5$", "bmssp_o_6": "$o_6$",
}


def plot_optimization_waterfall(data: Data, out_dir: Path, *, suffix: str = ""):
    size = 1048576
    source_key = "ef8"
    baseline = "dijkstra_opt"

    ratios, errs_lo, errs_hi, tick_labels, colors = [], [], [], [], []
    for algo in OPT_ALGOS:
        r = data.ratio(algo, baseline, size, source_key)
        if r is None:
            continue
        log_se = data.ratio_log_se(algo, baseline, size, source_key)
        if log_se is not None:
            z = 1.96
            lo = r - r * math.exp(-z * log_se)
            hi = r * math.exp(z * log_se) - r
        else:
            lo = hi = 0.0
        ratios.append(r)
        errs_lo.append(lo)
        errs_hi.append(hi)
        tick_labels.append(_OPT_SHORT[algo])
        colors.append(ALGO_STYLE[algo]["color"])

    if not ratios:
        print("  No data for optimization progression")
        return

    xs = list(range(len(ratios)))
    y_span = max(ratios) - min(ratios) or max(ratios) * 0.05

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    ax.plot(xs, ratios, color=KTH["office_80"], linewidth=1.2, zorder=2)
    ax.errorbar(xs, ratios, yerr=[errs_lo, errs_hi],
                fmt="none", ecolor=KTH["office_80"], elinewidth=1.0, capsize=3, zorder=3)
    for x, r, c in zip(xs, ratios, colors):
        ax.scatter([x], [r], color=c, s=60, zorder=4)

    for x, r in zip(xs, ratios):
        ax.text(x, r + y_span * 0.10, f"{r:.2f}x",
                ha="center", va="bottom", fontsize=7.5, color=KTH["office"])

    ax.text(0.99, 0.03,
            "Dijkstra (opt) = 1.00x (off-axis; lower is better for BMSSP)",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
            color=KTH["office"])

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Optimization round")
    ax.set_ylabel("BMSSP / Dijkstra (opt) ratio")

    exp = int(math.log2(size))
    title = f"BMSSP/Dijkstra ratio per optimization round ($n = 2^{{{exp}}}$, $m/n=8$)"
    if suffix:
        title = f"{title} — {data.hw_label}"
    ax.set_title(title, fontsize=10)

    ax.set_ylim(min(ratios) - y_span * 0.6, max(ratios) + y_span * 0.6)
    ax.set_xlim(-0.5, len(xs) - 0.5)

    ax.grid(True, which="major", axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out_dir / f"optimization_progression{suffix}.pdf", bbox_inches="tight")


def _draw_opt_ratio(ax, ratio_fn, common_sizes_fn, source_key: str = "ef8", ci_fn=None):
    baseline = "dijkstra_opt"
    sizes = common_sizes_fn([baseline] + OPT_ALGOS, source_key)
    for algo in OPT_ALGOS:
        xs, ys = _collect_ratio(ratio_fn, algo, baseline, sizes, source_key)
        if ys:
            if ci_fn is not None:
                ci_pts = [(n, ci_fn(algo, baseline, n, source_key)) for n in xs]
                xs_ci = [n for n, c in ci_pts if c is not None]
                lo_ci = [c[0] for _, c in ci_pts if c is not None]
                hi_ci = [c[1] for _, c in ci_pts if c is not None]
                if xs_ci:
                    ax.fill_between(xs_ci, lo_ci, hi_ci,
                                    alpha=ALGO_STYLE[algo]["alpha"] * 0.18,
                                    color=ALGO_STYLE[algo]["color"], zorder=1)
            ax.plot(xs, ys, label=ALGO_DISPLAY[algo], markersize=5, **ALGO_STYLE[algo])
    ax.axhline(y=1.0, color=KTH["black"], linewidth=1.5, linestyle="--", label="Dijkstra (opt)", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vertices $n$")
    ax.set_ylabel("Time ratio vs Dijkstra (opt)")
    _setup_xaxis(ax, sizes)
    # Show the 1.5x-10x band cleanly: explicit log ticks at integer ratios.
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.5, 2, 3, 5, 7, 10, 15, 20]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, which="major", alpha=0.3, linestyle="--")


def plot_optimization_ratio(data: Data, out_dir: Path, *, suffix: str = ""):
    source_key = "ef8"
    baseline = "dijkstra_opt"
    sizes = data.common_sizes([baseline] + OPT_ALGOS, source_key)
    if not sizes:
        print("  No common sizes for optimization ratio plot")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _draw_opt_ratio(ax, data.ratio, data.common_sizes, source_key)
    if suffix:
        ax.set_title(data.hw_label, fontsize=9)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / f"optimization_ratio{suffix}.pdf")


def plot_optimization_ratio_merged_single(multi: MultiData, out_dir: Path):
    source_key = "ef8"
    baseline = "dijkstra_opt"
    sizes = multi.common_sizes([baseline] + OPT_ALGOS, source_key)
    if not sizes:
        print("  No data for merged single optimization ratio plot")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _draw_opt_ratio(ax, multi.merged_ratio, multi.common_sizes, source_key, ci_fn=multi.merged_ratio_ci)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / "optimization_ratio.pdf")


def plot_optimization_ratio_multi(multi: MultiData, out_dir: Path):
    source_key = "ef8"
    baseline = "dijkstra_opt"
    sizes = multi.common_sizes([baseline] + OPT_ALGOS, source_key)
    if not sizes:
        print("  No data for multi optimization ratio")
        return

    ncols = 1 + len(multi.datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.4), squeeze=False, sharey=True)

    _draw_opt_ratio(axes[0][0], multi.merged_ratio, multi.common_sizes, source_key)
    axes[0][0].set_title("Merged (geometric mean)")

    for col, d in enumerate(multi.datasets, start=1):
        _draw_opt_ratio(axes[0][col], d.ratio, d.common_sizes, source_key)
        axes[0][col].set_title(d.hw_short)
        axes[0][col].set_ylabel("")

    y_max = max(ax.get_ylim()[1] for ax in axes[0])
    y_min = min(ax.get_ylim()[0] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(y_min, y_max)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Optimization progression (ratio)", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _save(fig, out_dir / "optimization_ratio_merged.pdf", bbox_inches="tight")


OPT_CHANGES = {
    "bmssp_base": "Baseline (unoptimized)",
    "bmssp_o_1": "Recursion elimination",
    "bmssp_o_2": "HashMap $\\to$ Vec; unsafe indexing",
    "bmssp_o_3": "Cache packing (partial revert)",
    "bmssp_o_4": "Pull path reduction",
    "bmssp_o_5": "$\\sqrt{m}$ buckets; u16 hops; u128 cmp",
    "bmssp_o_6": "Unsafe relax in base case",
}


def generate_optimization_ratio_table(data: Data, out_dir: Path, *, suffix: str = ""):
    source_key = "ef8"
    baseline = "dijkstra_opt"
    target_sizes = [1048576, 16777216]

    base_times = {n: t for n in target_sizes if (t := data.mean(baseline, n, source_key)) is not None}
    if not base_times:
        print("  No baseline data for optimization ratio table")
        return

    body = []
    for algo in OPT_ALGOS:
        cols = []
        for n in target_sizes:
            t = data.mean(algo, n, source_key)
            bt = base_times.get(n)
            cols.append(f"{t / bt:.2f}" if t is not None and bt is not None and bt > 0 else "{--}")
        body.append(f"    {ALGO_DISPLAY[algo]} & {OPT_CHANGES.get(algo, '')} & {cols[0]} & {cols[1]} \\\\")

    label_suffix = f":{suffix.lstrip('_')}" if suffix else ""
    caption = (
        f"Ratio of each BMSSP variant's runtime to optimized Dijkstra on Erd\\H{{o}}s--R\\'enyi graphs at $m/n = 8$ on {data.hw_label_tex}." if suffix
        else r"Ratio of each BMSSP variant's runtime to optimized Dijkstra on Erd\H{o}s--R\'enyi graphs at $m/n = 8$ ($m$ = edges, $n$ = vertices)."
    )
    lines = _latex_table(
        caption,
        f"tab:optratio{label_suffix}", "l l S[table-format=1.2] S[table-format=1.2]",
        r"{Variant} & {Key change} & {Ratio ($n{=}2^{20}$)} & {Ratio ($n{=}2^{24}$)}",
        body,
    )
    _write_tex(out_dir / f"optimization_ratio_table{suffix}.tex", lines)


def generate_optimization_ratio_table_merged(multi: MultiData, out_dir: Path):
    source_key = "ef8"
    baseline = "dijkstra_opt"
    target_sizes = [1048576]

    size_cols = []
    col_specs = ["l", "l"]
    for n in target_sizes:
        exp = int(math.log2(n))
        for d in multi.datasets:
            size_cols.append(f"{{R {d.hw_short} ($2^{{{exp}}}$)}}")
            col_specs.append("S[table-format=1.2]")
        size_cols.append(f"{{Merged ($2^{{{exp}}}$)}}")
        col_specs.append("S[table-format=1.2]")
        size_cols.append(f"{{95\\% CI ($2^{{{exp}}}$)}}")
        col_specs.append("c")

    col_spec = " ".join(col_specs)
    header = r"{Variant} & {Key change} & " + " & ".join(size_cols)

    body = []
    for algo in OPT_ALGOS:
        cols = []
        for n in target_sizes:
            for d in multi.datasets:
                r = d.ratio(algo, baseline, n, source_key)
                cols.append(f"{r:.2f}" if r is not None else "{--}")
            mr = multi.merged_ratio(algo, baseline, n, source_key)
            cols.append(f"{mr:.2f}" if mr is not None else "{--}")
            ci = multi.merged_ratio_ci(algo, baseline, n, source_key)
            cols.append(f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci is not None else "{--}")
        row = " & ".join(cols)
        body.append(f"    {ALGO_DISPLAY[algo]} & {OPT_CHANGES.get(algo, '')} & {row} \\\\")

    lines = _latex_table(
        r"Ratio of each BMSSP variant's runtime to optimized Dijkstra on Erd\H{o}s--R\'enyi graphs at $m/n = 8$ ($n = 2^{20}$). Merged is the geometric mean of per-hardware ratios; 95\% CI propagated from per-platform Criterion bootstrap intervals via the delta method on log-ratios.",
        "tab:optratio:merged", col_spec, header, body,
        resizebox=True,
    )
    _write_tex(out_dir / "optimization_ratio_table_merged.tex", lines)


def plot_memory_baseline_vs_o6(data: Data, out_dir: Path, *, suffix: str = ""):
    source_key = "ef8"
    algo_ls = {"dijkstra_opt": "--", "bmssp_base": "-", "bmssp_o_6": "-"}

    sizes = data.common_sizes(MEMORY_PLOT_ALGOS, source_key)
    if not sizes:
        print("  No common sizes for memory comparison")
        return

    # Two panels: left = log-log absolute scaling (the headline shape claim,
    # linear-in-n holds across four decades), right = linear ratio over the
    # same x-axis. The single log-log absolute panel was misleading because
    # the 5-22% memory premium reads as visual coincidence on a 4-decade
    # y-axis; the ratio panel makes the gap legible.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    ax_abs = axes[0][0]
    ax_ratio = axes[0][1]

    for algo in MEMORY_PLOT_ALGOS:
        _plot_memory_line(ax_abs, data, algo, sizes, source_key, linestyle=algo_ls[algo])

    ax_abs.set_xscale("log")
    ax_abs.set_yscale("log")
    ax_abs.set_xlabel("Vertices $n$")
    ax_abs.set_ylabel("Peak memory (MiB)")
    ax_abs.set_title("Absolute (log--log)", fontsize=9)
    _setup_xaxis(ax_abs, sizes)
    ax_abs.grid(True, which="major", alpha=0.3, linestyle="--")
    ax_abs.legend(loc="upper left", framealpha=0.9, fontsize=8)

    xs_ratio: list[int] = []
    ys_o6: list[float] = []
    ys_base: list[float] = []
    for n in sizes:
        d_mem = data.memory("dijkstra_opt", n, source_key)
        b_mem = data.memory("bmssp_base", n, source_key)
        o6_mem = data.memory("bmssp_o_6", n, source_key)
        if d_mem and o6_mem:
            xs_ratio.append(n)
            ys_o6.append(o6_mem / d_mem)
            ys_base.append((b_mem / d_mem) if b_mem else float("nan"))

    if xs_ratio:
        ax_ratio.plot(xs_ratio, ys_base, label="BMSSP (base) / Dijkstra (opt)",
                      color=ALGO_STYLE["bmssp_base"]["color"], marker="o", linewidth=1.8)
        ax_ratio.plot(xs_ratio, ys_o6, label="BMSSP $o_6$ / Dijkstra (opt)",
                      color=ALGO_STYLE["bmssp_o_6"]["color"], marker="X", linewidth=1.8)
        ax_ratio.axhline(y=1.0, color=KTH["black"], linewidth=0.8, linestyle=":", alpha=0.6)
        ax_ratio.set_xscale("log")
        ax_ratio.set_xlabel("Vertices $n$")
        ax_ratio.set_ylabel("Memory ratio (linear)")
        ax_ratio.set_title("Ratio (linear)", fontsize=9)
        _setup_xaxis(ax_ratio, sizes)
        ax_ratio.set_ylim(0.95, max(max(ys_o6, default=1.05), max(ys_base, default=1.05)) * 1.02)
        ax_ratio.grid(True, which="major", alpha=0.3, linestyle="--")
        ax_ratio.legend(loc="upper right", framealpha=0.9, fontsize=8)

    if suffix:
        fig.suptitle(data.hw_label, fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / f"memory_scaling_comparison{suffix}.pdf")



def _hw_suffix(index: int) -> str:
    return f"_hw{index}"


def _run_single(data: Data, out_dir: Path, suffix: str = "", index: int | None = None):
    multi_mode = index is not None
    print(f"\n--- Per-hardware: {data.hw_label} (suffix={suffix!r}) ---")

    if not multi_mode:
        print("=== Edge-factor plots ===")
        for ef in EDGE_FACTORS:
            plot_absolute(data, ef, out_dir, suffix=suffix)
            plot_ratio(data, ef, out_dir, suffix=suffix)

        print("=== Topology group plots ===")
        for group in TOPOLOGY_GROUPS:
            plot_topology_group(data, group, out_dir, suffix=suffix)

    if not multi_mode or index == 0:
        print("=== Cross-topology comparison ===")
        plot_cross_topology_comparison(data, out_dir, suffix=suffix)

    if not multi_mode:
        print("=== Cross-topology ratio ===")
        plot_cross_topology_ratio(data, out_dir, suffix=suffix)

        print("=== Memory plots ===")
        plot_memory_scaling(data, out_dir, suffix=suffix)
        plot_memory_ratio(data, out_dir, suffix=suffix)
        generate_memory_table(data, out_dir, suffix=suffix)

    print("=== Summary table ===")
    generate_summary_table(data, out_dir, suffix=suffix)

    print("=== Per-topology summary table ===")
    generate_topology_summary_table(data, out_dir, suffix=suffix)

    if not multi_mode or index == 1:
        print("=== Optimization waterfall ===")
        plot_optimization_waterfall(data, out_dir, suffix=suffix)

    if not multi_mode:
        print("=== Optimization ratio ===")
        plot_optimization_ratio(data, out_dir, suffix=suffix)
        generate_optimization_ratio_table(data, out_dir, suffix=suffix)

        print("=== Memory baseline vs o6 ===")
        plot_memory_baseline_vs_o6(data, out_dir, suffix=suffix)


def _run_multi(multi: MultiData, out_dir: Path):
    print("\n--- Multi-hardware merged plots ---")

    print("=== Topology group (merged) ===")
    for group in TOPOLOGY_GROUPS:
        plot_topology_group_merged(multi, group, out_dir)

    print("=== Cross-topology ratio (multi) ===")
    plot_cross_topology_ratio_multi(multi, out_dir)

    print("=== Memory ratio (multi) ===")
    plot_memory_ratio_multi(multi, out_dir)

    print("=== Summary table (merged) ===")
    generate_summary_table_merged(multi, out_dir)

    print("=== Per-topology summary table (merged) ===")
    generate_topology_summary_table_merged(multi, out_dir)

    print("=== Optimization ratio (multi) ===")
    plot_optimization_ratio_merged_single(multi, out_dir)
    generate_optimization_ratio_table_merged(multi, out_dir)

    print("=== Canonical single-platform plots (hw0) ===")
    plot_optimization_waterfall(multi.datasets[0], out_dir)
    plot_memory_baseline_vs_o6(multi.datasets[0], out_dir)
    generate_memory_table(multi.datasets[0], out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis plots from benchmark results JSON")
    parser.add_argument("results", type=Path, nargs="+",
                        help="One or more bench/results/run_*.json files (multiple = multi-hardware comparison)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    for p in args.results:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            raise SystemExit(1)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [Data(p) for p in args.results]

    if len(datasets) == 1:
        _run_single(datasets[0], out_dir)
    else:
        for i, d in enumerate(datasets):
            _run_single(d, out_dir, suffix=_hw_suffix(i), index=i)
        multi = MultiData(datasets)
        _run_multi(multi, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
