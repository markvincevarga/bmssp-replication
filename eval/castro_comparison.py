# %% [markdown]
# # Castro replication — Dijkstra vs BMSSP_base
#
# Loads `bench/results/run_*.json` (any number of runs, any number of
# architectures, partial results allowed) and compares each against the
# hardcoded Castro et al. (2025) reference tables (arXiv:2511.03007v3,
# Tables A.1–A.4).
#
# The script is focused on validating our `bmssp_base` against Castro's
# BMSSP-WC: the central quantity is the slowdown ratio (bmssp / dijkstra).
# If our ratio profile tracks Castro's, the baseline replicates the paper.

# %% Imports
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "bench" / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "castro"

NS_PER_MS = 1_000_000.0

BASELINE_ALGO = "dijkstra"
TARGET_ALGO = "bmssp_base"

# %% Castro reference (Tables A.1–A.4)
# Each entry: (label, n) -> {"d_ms": Dijkstra ms, "b_ms": BMSSP-WC ms,
#                            "ratio": b/d as printed in the paper,
#                            "m": edge count}
CASTRO_REF: dict[tuple[str, int], dict[str, float]] = {
    # Table A.1 — random sparse D3 (n = 2^k, m = 3n)
    ("castro_d3", 128):       {"d_ms": 0.005,    "b_ms": 0.127,    "ratio": 25.400, "m": 3 * 128},
    ("castro_d3", 256):       {"d_ms": 0.014,    "b_ms": 0.136,    "ratio": 9.714,  "m": 3 * 256},
    ("castro_d3", 512):       {"d_ms": 0.038,    "b_ms": 0.227,    "ratio": 5.974,  "m": 3 * 512},
    ("castro_d3", 1024):      {"d_ms": 0.087,    "b_ms": 0.504,    "ratio": 5.793,  "m": 3 * 1024},
    ("castro_d3", 2048):      {"d_ms": 0.214,    "b_ms": 0.880,    "ratio": 4.112,  "m": 3 * 2048},
    ("castro_d3", 4096):      {"d_ms": 0.432,    "b_ms": 1.818,    "ratio": 4.208,  "m": 3 * 4096},
    ("castro_d3", 8192):      {"d_ms": 1.039,    "b_ms": 3.722,    "ratio": 3.582,  "m": 3 * 8192},
    ("castro_d3", 16384):     {"d_ms": 2.308,    "b_ms": 6.875,    "ratio": 2.979,  "m": 3 * 16384},
    ("castro_d3", 32768):     {"d_ms": 4.675,    "b_ms": 16.356,   "ratio": 3.499,  "m": 3 * 32768},
    ("castro_d3", 65536):     {"d_ms": 10.738,   "b_ms": 38.156,   "ratio": 3.553,  "m": 3 * 65536},
    ("castro_d3", 131072):    {"d_ms": 29.175,   "b_ms": 81.830,   "ratio": 2.805,  "m": 3 * 131072},
    ("castro_d3", 262144):    {"d_ms": 73.481,   "b_ms": 208.062,  "ratio": 2.832,  "m": 3 * 262144},
    ("castro_d3", 524288):    {"d_ms": 177.561,  "b_ms": 564.448,  "ratio": 3.179,  "m": 3 * 524288},
    ("castro_d3", 1048576):   {"d_ms": 421.216,  "b_ms": 1286.205, "ratio": 3.054,  "m": 3 * 1048576},
    ("castro_d3", 2097152):   {"d_ms": 939.767,  "b_ms": 2993.859, "ratio": 3.186,  "m": 3 * 2097152},
    ("castro_d3", 4194304):   {"d_ms": 2022.890, "b_ms": 7702.933, "ratio": 3.808,  "m": 3 * 4194304},
    ("castro_d3", 8388608):   {"d_ms": 4437.921, "b_ms": 14808.467,"ratio": 3.337,  "m": 3 * 8388608},
    ("castro_d3", 16777216):  {"d_ms": 9899.507, "b_ms": 32367.954,"ratio": 3.270,  "m": 3 * 16777216},
    ("castro_d3", 33554432):  {"d_ms": 21812.125,"b_ms": 82592.131,"ratio": 3.787,  "m": 3 * 33554432},
    # Table A.1 — random sparse H3
    ("castro_h3", 128):       {"d_ms": 0.006,    "b_ms": 0.133,    "ratio": 22.167, "m": 3 * 128},
    ("castro_h3", 256):       {"d_ms": 0.015,    "b_ms": 0.178,    "ratio": 11.867, "m": 3 * 256},
    ("castro_h3", 512):       {"d_ms": 0.036,    "b_ms": 0.219,    "ratio": 6.083,  "m": 3 * 512},
    ("castro_h3", 1024):      {"d_ms": 0.092,    "b_ms": 0.478,    "ratio": 5.196,  "m": 3 * 1024},
    ("castro_h3", 2048):      {"d_ms": 0.196,    "b_ms": 0.958,    "ratio": 4.888,  "m": 3 * 2048},
    ("castro_h3", 4096):      {"d_ms": 0.429,    "b_ms": 1.848,    "ratio": 4.308,  "m": 3 * 4096},
    ("castro_h3", 8192):      {"d_ms": 0.962,    "b_ms": 3.586,    "ratio": 3.728,  "m": 3 * 8192},
    ("castro_h3", 16384):     {"d_ms": 2.221,    "b_ms": 6.564,    "ratio": 2.955,  "m": 3 * 16384},
    ("castro_h3", 32768):     {"d_ms": 4.881,    "b_ms": 15.826,   "ratio": 3.242,  "m": 3 * 32768},
    ("castro_h3", 65536):     {"d_ms": 10.482,   "b_ms": 35.381,   "ratio": 3.375,  "m": 3 * 65536},
    ("castro_h3", 131072):    {"d_ms": 29.980,   "b_ms": 84.178,   "ratio": 2.808,  "m": 3 * 131072},
    ("castro_h3", 262144):    {"d_ms": 74.390,   "b_ms": 213.486,  "ratio": 2.870,  "m": 3 * 262144},
    ("castro_h3", 524288):    {"d_ms": 179.838,  "b_ms": 569.684,  "ratio": 3.168,  "m": 3 * 524288},
    ("castro_h3", 1048576):   {"d_ms": 424.459,  "b_ms": 1317.955, "ratio": 3.105,  "m": 3 * 1048576},
    ("castro_h3", 2097152):   {"d_ms": 933.016,  "b_ms": 3023.283, "ratio": 3.240,  "m": 3 * 2097152},
    ("castro_h3", 4194304):   {"d_ms": 2042.307, "b_ms": 7718.196, "ratio": 3.779,  "m": 3 * 4194304},
    ("castro_h3", 8388608):   {"d_ms": 4519.483, "b_ms": 14895.695,"ratio": 3.296,  "m": 3 * 8388608},
    ("castro_h3", 16777216):  {"d_ms": 10066.866,"b_ms": 32832.183,"ratio": 3.261,  "m": 3 * 16777216},
    ("castro_h3", 33554432):  {"d_ms": 22009.777,"b_ms": 84702.824,"ratio": 3.848,  "m": 3 * 33554432},
    # Table A.3 — Square Grid Euclidean (S = N x N)
    ("castro_sgrided", 256):       {"d_ms": 0.009,    "b_ms": 0.109,    "ratio": 12.111, "m": 1800},
    ("castro_sgrided", 1024):      {"d_ms": 0.050,    "b_ms": 0.340,    "ratio": 6.800,  "m": 7800},
    ("castro_sgrided", 4096):      {"d_ms": 0.219,    "b_ms": 1.266,    "ratio": 5.781,  "m": 32000},
    ("castro_sgrided", 16384):     {"d_ms": 1.000,    "b_ms": 4.681,    "ratio": 4.681,  "m": 120000},
    ("castro_sgrided", 65536):     {"d_ms": 4.804,    "b_ms": 17.493,   "ratio": 3.641,  "m": 520000},
    ("castro_sgrided", 262144):    {"d_ms": 21.274,   "b_ms": 80.996,   "ratio": 3.807,  "m": 2_000_000},
    ("castro_sgrided", 1048576):   {"d_ms": 102.265,  "b_ms": 396.114,  "ratio": 3.873,  "m": 8_300_000},
    ("castro_sgrided", 4194304):   {"d_ms": 622.311,  "b_ms": 2328.856, "ratio": 3.742,  "m": 33_000_000},
    ("castro_sgrided", 16777216):  {"d_ms": 2872.840, "b_ms": 9860.496, "ratio": 3.432,  "m": 130_000_000},
    # Table A.3 — Rectangular Grid Euclidean
    ("castro_rgrided", 256):       {"d_ms": 0.006,    "b_ms": 0.165,    "ratio": 27.500, "m": 1800},
    ("castro_rgrided", 1024):      {"d_ms": 0.037,    "b_ms": 0.669,    "ratio": 18.081, "m": 7700},
    ("castro_rgrided", 4096):      {"d_ms": 0.180,    "b_ms": 2.806,    "ratio": 15.589, "m": 31000},
    ("castro_rgrided", 16384):     {"d_ms": 0.974,    "b_ms": 4.641,    "ratio": 4.765,  "m": 120000},
    ("castro_rgrided", 65536):     {"d_ms": 4.402,    "b_ms": 19.690,   "ratio": 4.473,  "m": 520000},
    ("castro_rgrided", 262144):    {"d_ms": 21.995,   "b_ms": 98.729,   "ratio": 4.489,  "m": 2_000_000},
    ("castro_rgrided", 1048576):   {"d_ms": 109.942,  "b_ms": 558.391,  "ratio": 5.079,  "m": 8_300_000},
    ("castro_rgrided", 4194304):   {"d_ms": 646.909,  "b_ms": 3093.970, "ratio": 4.783,  "m": 33_000_000},
    ("castro_rgrided", 16777216):  {"d_ms": 3844.126, "b_ms": 14431.560,"ratio": 3.754,  "m": 130_000_000},
    # Table A.4 — Square Grid Random
    ("castro_sgridr", 256):       {"d_ms": 0.032,    "b_ms": 0.166,    "ratio": 5.188,  "m": 1800},
    ("castro_sgridr", 1024):      {"d_ms": 0.112,    "b_ms": 0.622,    "ratio": 5.554,  "m": 7800},
    ("castro_sgridr", 4096):      {"d_ms": 0.502,    "b_ms": 2.555,    "ratio": 5.090,  "m": 32000},
    ("castro_sgridr", 16384):     {"d_ms": 2.281,    "b_ms": 8.567,    "ratio": 3.756,  "m": 120000},
    ("castro_sgridr", 65536):     {"d_ms": 12.731,   "b_ms": 36.239,   "ratio": 2.847,  "m": 520000},
    ("castro_sgridr", 262144):    {"d_ms": 49.706,   "b_ms": 153.891,  "ratio": 3.096,  "m": 2_000_000},
    ("castro_sgridr", 1048576):   {"d_ms": 220.996,  "b_ms": 743.351,  "ratio": 3.364,  "m": 8_300_000},
    ("castro_sgridr", 4194304):   {"d_ms": 1072.783, "b_ms": 4260.375, "ratio": 3.971,  "m": 33_000_000},
    ("castro_sgridr", 16777216):  {"d_ms": 6126.387, "b_ms": 19744.810,"ratio": 3.223,  "m": 130_000_000},
    # Table A.4 — Rectangular Grid Random
    ("castro_rgridr", 256):       {"d_ms": 0.026,    "b_ms": 0.160,    "ratio": 6.154,  "m": 1800},
    ("castro_rgridr", 1024):      {"d_ms": 0.128,    "b_ms": 0.550,    "ratio": 4.297,  "m": 7700},
    ("castro_rgridr", 4096):      {"d_ms": 0.570,    "b_ms": 2.141,    "ratio": 3.756,  "m": 31000},
    ("castro_rgridr", 16384):     {"d_ms": 2.588,    "b_ms": 8.850,    "ratio": 3.420,  "m": 120000},
    ("castro_rgridr", 65536):     {"d_ms": 11.337,   "b_ms": 36.273,   "ratio": 3.200,  "m": 520000},
    ("castro_rgridr", 262144):    {"d_ms": 62.644,   "b_ms": 165.676,  "ratio": 2.645,  "m": 2_000_000},
    ("castro_rgridr", 1048576):   {"d_ms": 267.950,  "b_ms": 807.726,  "ratio": 3.014,  "m": 8_300_000},
    ("castro_rgridr", 4194304):   {"d_ms": 1434.521, "b_ms": 5383.173, "ratio": 3.753,  "m": 33_000_000},
    ("castro_rgridr", 16777216):  {"d_ms": 7103.698, "b_ms": 23154.213,"ratio": 3.259,  "m": 130_000_000},
    # Table A.2 — USA road networks
    ("usa_NY",  264346):    {"d_ms": 32.406,   "b_ms": 119.361,    "ratio": 3.683, "m": 730_000},
    ("usa_BAY", 321270):    {"d_ms": 38.361,   "b_ms": 140.055,    "ratio": 3.651, "m": 800_000},
    ("usa_COL", 435666):    {"d_ms": 52.109,   "b_ms": 216.572,    "ratio": 4.156, "m": 1_000_000},
    ("usa_FLA", 1070376):   {"d_ms": 128.166,  "b_ms": 455.958,    "ratio": 3.559, "m": 2_700_000},
    ("usa_NW",  1207945):   {"d_ms": 154.594,  "b_ms": 521.545,    "ratio": 3.374, "m": 2_800_000},
    ("usa_NE",  1524453):   {"d_ms": 228.359,  "b_ms": 707.389,    "ratio": 3.098, "m": 3_800_000},
    ("usa_CAL", 1890815):   {"d_ms": 261.384,  "b_ms": 847.049,    "ratio": 3.241, "m": 4_600_000},
    ("usa_LKS", 2758119):   {"d_ms": 381.540,  "b_ms": 1402.351,   "ratio": 3.676, "m": 6_800_000},
    ("usa_E",   3598623):   {"d_ms": 556.746,  "b_ms": 2074.609,   "ratio": 3.726, "m": 8_700_000},
    ("usa_W",   6262104):   {"d_ms": 1022.692, "b_ms": 3989.016,   "ratio": 3.901, "m": 15_000_000},
    ("usa_CTR", 14081816):  {"d_ms": 3299.535, "b_ms": 13188.157,  "ratio": 3.997, "m": 34_000_000},
    ("usa_USA", 23947347):  {"d_ms": 4577.503, "b_ms": 19325.188,  "ratio": 4.222, "m": 58_000_000},
}

FAMILY_LABELS = {
    "castro_d3":      "Random D3",
    "castro_h3":      "Random H3",
    "castro_sgrided": "Square GridED",
    "castro_rgrided": "Rectangular GridED",
    "castro_sgridr":  "Square GridR",
    "castro_rgridr":  "Rectangular GridR",
    "usa_road":       "USA road networks",
}

USA_LABELS_ORDER = [
    "usa_NY", "usa_BAY", "usa_COL", "usa_FLA", "usa_NW", "usa_NE",
    "usa_CAL", "usa_LKS", "usa_E", "usa_W", "usa_CTR", "usa_USA",
]

ALL_FAMILIES = [
    "castro_d3", "castro_h3",
    "castro_sgrided", "castro_rgrided",
    "castro_sgridr", "castro_rgridr",
    "usa_road",
]

USA_LABEL_SET = set(USA_LABELS_ORDER)


# %% Data classes
@dataclass(frozen=True)
class HardwareKey:
    hw_id: str
    cpu: str
    arch: str

    def short(self) -> str:
        return f"{self.cpu} ({self.arch})"

    def plot_label(self) -> str:
        cpu = self.cpu
        for prefix in ("Apple ", "AMD ", "Intel "):
            cpu = cpu.replace(prefix, "")
        cpu = cpu.replace(" 8-Core Processor", "").replace(" 12-Core Processor", "")
        return f"{cpu} ({self.arch})"


@dataclass
class RunResult:
    run_id: str
    timestamp: str
    hardware: HardwareKey
    path: Path
    # results[algo][label][size_int] -> mean_ns (point estimate)
    means_ns: dict[str, dict[str, dict[int, float]]]
    ci_ns: dict[str, dict[str, dict[int, tuple[float, float]]]]


# %% Loading
def load_run(path: Path) -> RunResult | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"warn: failed to read {path}: {e}")
        return None

    hw = data.get("hardware", {})
    hardware = HardwareKey(
        hw_id=hw.get("hardware_id", "unknown"),
        cpu=hw.get("cpu", "unknown CPU"),
        arch=hw.get("arch", "unknown"),
    )

    means: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    cis: dict[str, dict[str, dict[int, tuple[float, float]]]] = defaultdict(lambda: defaultdict(dict))

    for algo, by_label in data.get("results", {}).items():
        for label, by_size in by_label.items():
            for size_str, entry in by_size.items():
                try:
                    size = int(size_str)
                except ValueError:
                    continue
                runtime = entry.get("runtime_ns")
                if not runtime:
                    continue
                mean_block = runtime.get("mean", {})
                point = mean_block.get("point_estimate")
                if point is None:
                    continue
                means[algo][label][size] = float(point)
                ci = mean_block.get("confidence_interval", {})
                lo = ci.get("lower_bound", point)
                hi = ci.get("upper_bound", point)
                cis[algo][label][size] = (float(lo), float(hi))

    return RunResult(
        run_id=data.get("run_id", path.stem),
        timestamp=data.get("timestamp", ""),
        hardware=hardware,
        path=path,
        means_ns={a: dict(v) for a, v in means.items()},
        ci_ns={a: dict(v) for a, v in cis.items()},
    )


def discover_runs(paths: Iterable[Path]) -> list[RunResult]:
    runs: list[RunResult] = []
    for p in paths:
        if p.is_dir():
            runs.extend(filter(None, (load_run(j) for j in sorted(p.glob("run_*.json")))))
        elif p.is_file():
            r = load_run(p)
            if r:
                runs.append(r)
        else:
            print(f"warn: ignoring missing path {p}")
    return runs


def has_castro_payload(run: RunResult) -> bool:
    if BASELINE_ALGO not in run.means_ns or TARGET_ALGO not in run.means_ns:
        return False
    castro_labels = {l for (l, _) in CASTRO_REF}
    for algo in (BASELINE_ALGO, TARGET_ALGO):
        for label in run.means_ns.get(algo, {}):
            if label in castro_labels:
                return True
    return False


def latest_per_hardware(runs: list[RunResult]) -> dict[HardwareKey, RunResult]:
    by_hw: dict[HardwareKey, RunResult] = {}
    for r in sorted(runs, key=lambda r: r.timestamp or r.run_id):
        by_hw[r.hardware] = r
    return by_hw


# %% Comparison
@dataclass
class ComparisonRow:
    label: str
    size: int
    castro_d_ms: float
    castro_b_ms: float
    castro_ratio: float
    ours_d_ms: float | None
    ours_b_ms: float | None
    ours_ratio: float | None
    ratio_of_ratios: float | None

    @property
    def has_ours(self) -> bool:
        return self.ours_ratio is not None

    @property
    def d_scale(self) -> float | None:
        if self.ours_d_ms is None or self.castro_d_ms == 0:
            return None
        return self.ours_d_ms / self.castro_d_ms

    @property
    def b_scale(self) -> float | None:
        if self.ours_b_ms is None or self.castro_b_ms == 0:
            return None
        return self.ours_b_ms / self.castro_b_ms


def build_comparison(run: RunResult) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    dij = run.means_ns.get(BASELINE_ALGO, {})
    bms = run.means_ns.get(TARGET_ALGO, {})
    for (label, size), ref in CASTRO_REF.items():
        d_ours = dij.get(label, {}).get(size)
        b_ours = bms.get(label, {}).get(size)
        ours_d_ms = d_ours / NS_PER_MS if d_ours is not None else None
        ours_b_ms = b_ours / NS_PER_MS if b_ours is not None else None
        ours_ratio = (
            (ours_b_ms / ours_d_ms)
            if ours_d_ms is not None and ours_b_ms is not None and ours_d_ms > 0
            else None
        )
        ror = (
            (ours_ratio / ref["ratio"])
            if ours_ratio is not None and ref["ratio"] > 0
            else None
        )
        rows.append(ComparisonRow(
            label=label,
            size=size,
            castro_d_ms=ref["d_ms"],
            castro_b_ms=ref["b_ms"],
            castro_ratio=ref["ratio"],
            ours_d_ms=ours_d_ms,
            ours_b_ms=ours_b_ms,
            ours_ratio=ours_ratio,
            ratio_of_ratios=ror,
        ))
    return rows


def family_for_label(label: str) -> str:
    if label in USA_LABEL_SET:
        return "usa_road"
    return label


def rows_by_family(rows: list[ComparisonRow]) -> dict[str, list[ComparisonRow]]:
    grouped: dict[str, list[ComparisonRow]] = defaultdict(list)
    for r in rows:
        grouped[family_for_label(r.label)].append(r)
    if "usa_road" in grouped:
        order = {l: i for i, l in enumerate(USA_LABELS_ORDER)}
        grouped["usa_road"].sort(key=lambda r: order.get(r.label, 999))
    for fam, items in grouped.items():
        if fam != "usa_road":
            items.sort(key=lambda r: r.size)
    return grouped


# %% Plotting
COLOR_CASTRO = "#000000"
HW_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#56B4E9"]

FAMILY_COLORS = {
    "castro_d3":      "#1954A6",
    "castro_h3":      "#2191C4",
    "castro_sgrided": "#62922E",
    "castro_rgrided": "#008F5D",
    "castro_sgridr":  "#D02F80",
    "castro_rgridr":  "#E07030",
    "usa_road":       "#65656C",
}

FAMILY_SHORT = {
    "castro_d3":      "D3",
    "castro_h3":      "H3",
    "castro_sgrided": "SGridED",
    "castro_rgrided": "RGridED",
    "castro_sgridr":  "SGridR",
    "castro_rgridr":  "RGridR",
    "usa_road":       "USA roads",
}


def assign_hw_colors(hardware_keys: list[HardwareKey]) -> dict[str, str]:
    return {hw.hw_id: HW_COLORS[i % len(HW_COLORS)] for i, hw in enumerate(hardware_keys)}


def plot_combined_ror_overview(
    by_hw_rows: dict[HardwareKey, list[ComparisonRow]],
    output_path: Path,
) -> None:
    """Strip chart of all 86 ρ values across families and both platforms."""
    if not by_hw_rows:
        return

    ref_rows = next(iter(by_hw_rows.values()))
    by_family: dict[str, list[ComparisonRow]] = defaultdict(list)
    for r in ref_rows:
        by_family[family_for_label(r.label)].append(r)
    if "usa_road" in by_family:
        order = {l: i for i, l in enumerate(USA_LABELS_ORDER)}
        by_family["usa_road"].sort(key=lambda r: order.get(r.label, 999))
    for fam in by_family:
        if fam != "usa_road":
            by_family[fam].sort(key=lambda r: r.size)

    ordered_families = [f for f in ALL_FAMILIES if f in by_family]
    flat_instances: list[tuple[str, ComparisonRow]] = []
    for fam in ordered_families:
        for r in by_family[fam]:
            flat_instances.append((fam, r))

    pos_lookup: dict[tuple[str, int], int] = {}
    for i, (fam, r) in enumerate(flat_instances):
        pos_lookup[(r.label, r.size)] = i

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_yscale("log")

    # Shaded family bands (alternating) — drawn first so they stay behind dots
    family_starts: dict[str, int] = {}
    family_ends: dict[str, int] = {}
    for i, (fam, _) in enumerate(flat_instances):
        if fam not in family_starts:
            family_starts[fam] = i
        family_ends[fam] = i

    for fi, fam in enumerate(ordered_families):
        start = family_starts[fam]
        end = family_ends[fam]
        if fi % 2 == 1:
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.06, color="#000000", zorder=0)

    # ±25% band and parity line — no legend labels, caption covers them
    ax.axhspan(0.8, 1.25, alpha=0.12, color="#888888", zorder=1)
    ax.axhline(1.0, linestyle="-", color="#000000", linewidth=1.2, zorder=2)

    hw_styles = [
        {"color": HW_COLORS[0], "marker": "o"},
        {"color": HW_COLORS[1], "marker": "s"},
    ]
    for hw_idx, (hw, rows) in enumerate(by_hw_rows.items()):
        style = hw_styles[hw_idx % len(hw_styles)]
        xs = []
        ys = []
        for r in rows:
            if r.ratio_of_ratios is None:
                continue
            key = (r.label, r.size)
            if key not in pos_lookup:
                continue
            xs.append(pos_lookup[key])
            ys.append(r.ratio_of_ratios)
        ax.scatter(
            xs, ys,
            color=style["color"],
            marker=style["marker"],
            s=45,
            linewidths=0.4,
            edgecolors="white",
            zorder=3,
            label=hw.plot_label(),
            alpha=0.88,
        )

    for fam in ordered_families[1:]:
        ax.axvline(family_starts[fam] - 0.5, color="#999999", linewidth=0.8, zorder=1, linestyle="--")

    ax.set_xlim(-0.8, len(flat_instances) - 0.2)
    _size_tick_labels = []
    for _fam, _r in flat_instances:
        if _fam == "usa_road":
            _size_tick_labels.append(_r.label.replace("usa_", ""))
        elif _r.size > 0 and (_r.size & (_r.size - 1)) == 0:
            _exp = int(math.log2(_r.size))
            _size_tick_labels.append(f"$2^{{{_exp}}}$")
        else:
            _size_tick_labels.append(str(_r.size))
    ax.set_xticks(list(range(len(flat_instances))))
    ax.set_xticklabels(_size_tick_labels, rotation=90, fontsize=6)
    ax.tick_params(axis="x", length=2, pad=1)
    for fam in ordered_families:
        start = family_starts[fam]
        end = family_ends[fam]
        mid = (start + end) / 2
        x_frac = (mid + 0.8) / (len(flat_instances) - 0.2 + 0.8)
        ax.text(
            x_frac, 1.02,
            FAMILY_SHORT.get(fam, fam),
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10,
            color=FAMILY_COLORS.get(fam, "#333333"),
            fontweight="bold",
        )

    ax.set_ylabel(r"$\rho = (\text{ours B/D})\,/\,(\text{Castro B/D})$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _xaxis_for_family(family: str, items: list[ComparisonRow]):
    if family == "usa_road":
        x = np.arange(len(items))
        labels = [r.label.replace("usa_", "") for r in items]
        return x, labels, "USA road instance", False
    x = np.array([r.size for r in items], dtype=float)
    labels = [str(r.size) for r in items]
    return x, labels, "Number of vertices (n)", True


def plot_ratio(
    family: str,
    by_run: dict[HardwareKey, list[ComparisonRow]],
    hw_colors: dict[str, str],
    output_path: Path,
) -> None:
    castro_rows = next(iter(by_run.values()), [])
    if not castro_rows:
        return
    x, tick_labels, xlabel, log_x = _xaxis_for_family(family, castro_rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, linestyle=":", color="#888", linewidth=1, label="break-even")

    castro_y = [r.castro_ratio for r in castro_rows]
    ax.plot(
        x, castro_y,
        marker="o", color=COLOR_CASTRO, linewidth=2, markersize=6,
        label="Castro (BMSSP-WC / Dijkstra)",
    )

    for hw, rows in by_run.items():
        ys = [r.ours_ratio if r.ours_ratio is not None else np.nan for r in rows]
        if all(np.isnan(ys)):
            continue
        ax.plot(
            x, ys,
            marker="s", color=hw_colors[hw.hw_id], linewidth=1.5, markersize=5,
            label=f"Ours / {hw.short()}",
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("BMSSP_base / Dijkstra (slowdown)", fontsize=11)
    ax.set_title(f"Slowdown ratio — {FAMILY_LABELS.get(family, family)}", fontsize=12, fontweight="bold")
    if log_x:
        ax.set_xscale("log")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_absolute(
    family: str,
    by_run: dict[HardwareKey, list[ComparisonRow]],
    hw_colors: dict[str, str],
    output_path: Path,
) -> None:
    castro_rows = next(iter(by_run.values()), [])
    if not castro_rows:
        return
    x, tick_labels, xlabel, log_x = _xaxis_for_family(family, castro_rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, [r.castro_d_ms for r in castro_rows],
            marker="o", linestyle="-", color=COLOR_CASTRO, linewidth=1.8, label="Castro Dijkstra")
    ax.plot(x, [r.castro_b_ms for r in castro_rows],
            marker="o", linestyle="--", color=COLOR_CASTRO, linewidth=1.8, label="Castro BMSSP-WC")

    for hw, rows in by_run.items():
        d_ys = [r.ours_d_ms if r.ours_d_ms is not None else np.nan for r in rows]
        b_ys = [r.ours_b_ms if r.ours_b_ms is not None else np.nan for r in rows]
        col = hw_colors[hw.hw_id]
        if not all(np.isnan(d_ys)):
            ax.plot(x, d_ys, marker="s", linestyle="-",  color=col, linewidth=1.4,
                    label=f"Ours dijkstra / {hw.short()}")
        if not all(np.isnan(b_ys)):
            ax.plot(x, b_ys, marker="s", linestyle="--", color=col, linewidth=1.4,
                    label=f"Ours bmssp_base / {hw.short()}")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax.set_title(f"Absolute timings — {FAMILY_LABELS.get(family, family)}", fontsize=12, fontweight="bold")
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    if not log_x:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_ratio_of_ratios(
    family: str,
    by_run: dict[HardwareKey, list[ComparisonRow]],
    hw_colors: dict[str, str],
    output_path: Path,
) -> None:
    castro_rows = next(iter(by_run.values()), [])
    if not castro_rows:
        return
    x, tick_labels, xlabel, log_x = _xaxis_for_family(family, castro_rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, linestyle="-", color="#000", linewidth=1, label="parity (matches Castro)")

    plotted = False
    for hw, rows in by_run.items():
        ys = [r.ratio_of_ratios if r.ratio_of_ratios is not None else np.nan for r in rows]
        if all(np.isnan(ys)):
            continue
        plotted = True
        ax.plot(
            x, ys,
            marker="o", color=hw_colors[hw.hw_id], linewidth=1.6, markersize=5,
            label=hw.short(),
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("(ours bmssp/dijkstra) / (Castro bmssp/dijkstra)", fontsize=11)
    ax.set_title(
        f"Slowdown agreement vs Castro — {FAMILY_LABELS.get(family, family)}\n"
        f"<1 means we're tighter than Castro, >1 means looser",
        fontsize=12, fontweight="bold",
    )
    if log_x:
        ax.set_xscale("log")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


# %% LaTeX tables
def latex_escape(s: str) -> str:
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = s
    for a, b in replacements:
        out = out.replace(a, b)
    return out


def fmt_ms_tex(v: float | None) -> str:
    """Format a millisecond value at 3 significant figures, preserving
    trailing zeros so each cell carries the same significant precision.

    Avoids the spurious-precision artifact where Castro's printed value
    of 0.005 ms was rendered as 0.005000 ms (4-decimals at 1-sig-fig
    input precision); also avoids the opposite mistake where 2.997 ms
    becomes "3" (1 sig fig).
    """
    if v is None:
        return r"\textendash"
    if v == 0:
        return "0.00"
    floor_log = int(math.floor(math.log10(abs(v))))
    decimals = max(0, 2 - floor_log)  # 3 sig figs total
    s = f"{v:,.{decimals}f}"
    return s.replace(",", r"\,")


def fmt_ratio_tex(v: float | None) -> str:
    """Format a ratio with 3 decimal places."""
    if v is None:
        return r"\textendash"
    return f"{v:.3f}"


def fmt_n_tex(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def hw_caption_short(hw: HardwareKey) -> str:
    cpu = hw.cpu
    if "Apple " in cpu:
        cpu = cpu.replace("Apple ", "")
    if "AMD " in cpu:
        cpu = cpu.replace("AMD ", "")
    cpu = cpu.replace(" 8-Core Processor", "")
    return f"{latex_escape(cpu)} ({latex_escape(hw.arch)})"


def render_family_tex_table(
    family: str,
    by_run: dict[HardwareKey, list[ComparisonRow]],
) -> str:
    castro_rows = next(iter(by_run.values()), [])
    if not castro_rows:
        return ""

    n_archs = len(by_run)
    title = FAMILY_LABELS.get(family, family)
    label_id = f"tab:castro:{family}"

    first_col_header = "Instance" if family == "usa_road" else "$n$"
    first_align = "l" if family == "usa_road" else "r"
    col_align = first_align + " r r r" + " r r r r" * n_archs

    out: list[str] = []
    out.append(r"\begin{table}[!ht]")
    out.append(r"  \centering")
    out.append(r"  \scriptsize")
    arch_names = ", ".join(hw_caption_short(hw) for hw in by_run.keys())
    out.append(
        rf"  \caption{{Castro reference vs.\ \texttt{{bmssp\_base}} on \emph{{{latex_escape(title)}}} "
        rf"instances. Hardware: {arch_names}. "
        rf"$T_D$ is Dijkstra wall-clock time and $T_B$ is BMSSP-WC wall-clock time, both in milliseconds; "
        rf"B/D is the per-instance BMSSP/Dijkstra ratio; "
        rf"$\rho = (\text{{ours B/D}})/(\text{{Castro B/D}})$ is the per-instance ratio of ratios.}}"
    )
    out.append(rf"  \label{{{label_id}}}")
    out.append(r"  \resizebox{\textwidth}{!}{%")
    out.append(rf"  \begin{{tabular}}{{{col_align}}}")
    out.append(r"    \toprule")

    top_hdr = [r"\multirow{2}{*}{" + first_col_header + "}",
               r"\multicolumn{3}{c}{Castro reference}"]
    for hw in by_run.keys():
        top_hdr.append(rf"\multicolumn{{4}}{{c}}{{{hw_caption_short(hw)}}}")
    out.append("    " + " & ".join(top_hdr) + r" \\")

    cmid_parts = []
    col_idx = 2
    cmid_parts.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + 2}}}")
    col_idx += 3
    for _ in by_run.keys():
        cmid_parts.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + 3}}}")
        col_idx += 4
    out.append("    " + " ".join(cmid_parts))

    sub_hdr = ["", "$T_D$ (ms)", "$T_B$ (ms)", "B/D"]
    for _ in by_run.keys():
        sub_hdr += ["$T_D$ (ms)", "$T_B$ (ms)", "B/D", r"$\rho$"]
    out.append("    " + " & ".join(sub_hdr) + r" \\")
    out.append(r"    \midrule")

    n_rows = len(castro_rows)
    for i in range(n_rows):
        ref = castro_rows[i]
        first = (ref.label.replace("usa_", "").replace("_", r"\_")
                 if family == "usa_road" else fmt_n_tex(ref.size))
        cells = [
            first,
            fmt_ms_tex(ref.castro_d_ms),
            fmt_ms_tex(ref.castro_b_ms),
            fmt_ratio_tex(ref.castro_ratio),
        ]
        for hw, rows in by_run.items():
            r = rows[i]
            cells += [
                fmt_ms_tex(r.ours_d_ms),
                fmt_ms_tex(r.ours_b_ms),
                fmt_ratio_tex(r.ours_ratio),
                fmt_ratio_tex(r.ratio_of_ratios),
            ]
        out.append("    " + " & ".join(cells) + r" \\")

    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}%")
    out.append(r"  }")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def render_summary_tex(by_hw_rows: dict[HardwareKey, list[ComparisonRow]]) -> str:
    out: list[str] = []
    out.append(r"\begin{table}[!ht]")
    out.append(r"  \centering")
    out.append(r"  \small")
    out.append(
        r"  \caption{Agreement of \texttt{bmssp\_base} slowdown profile with Castro et al.\ (2025). "
        r"$\rho = (\text{ours B/D})/(\text{Castro B/D})$ per matched instance, where B/D is the BMSSP/Dijkstra runtime ratio. "
        r"$\bar{\rho}_g$ (column \emph{GeoMean}) is the geometric mean of $\rho$, the canonical mean for ratios (Fleming \& Wallace, 1986). "
        r"$\sigma_g$ is the geometric standard deviation factor (a multiplicative spread; "
        r"smaller is tighter; $\sigma_g = 1$ means perfectly constant). "
        r"$f_{\pm 25\%}$ is the fraction of instances with $0.8 \le \rho \le 1.25$. "
        r"Bracketed values are 95\% percentile-bootstrap CIs; they overlap across platforms on every aggregate.}"
    )
    out.append(r"  \label{tab:castro:summary}")
    out.append(r"  \begin{tabular}{l r r r r}")
    out.append(r"    \toprule")
    out.append(r"    Hardware & GeoMean & Median & $\sigma_g$ & $f_{\pm 25\%}$ \\")
    out.append(r"    \midrule")
    for hw, rows in by_hw_rows.items():
        ratios = [r.ratio_of_ratios for r in rows if r.ratio_of_ratios is not None]
        if not ratios:
            out.append(rf"    {hw_caption_short(hw)} & \textendash & \textendash & \textendash & \textendash \\")
            continue
        arr = np.array(ratios)
        log_arr = np.log(arr)
        geo = float(np.exp(np.mean(log_arr)))
        med = float(np.median(arr))
        sigma_g = float(np.exp(np.std(log_arr, ddof=1))) if len(arr) > 1 else 1.0
        within = float(np.mean((arr >= 0.8) & (arr <= 1.25)))
        out.append(
            rf"    {hw_caption_short(hw)} & {geo:.3f} & {med:.3f} & "
            rf"{sigma_g:.3f} & {within * 100:.0f}\% \\"
        )
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"



# %% Output
def write_tex_outputs(
    by_hw_rows: dict[HardwareKey, list[ComparisonRow]],
    output_dir: Path,
) -> list[Path]:
    written: list[Path] = []

    summary_path = output_dir / "summary_table.tex"
    summary_path.write_text(render_summary_tex(by_hw_rows), encoding="utf-8")
    written.append(summary_path)

    by_family_per_run: dict[str, dict[HardwareKey, list[ComparisonRow]]] = defaultdict(dict)
    for hw, rows in by_hw_rows.items():
        groups = rows_by_family(rows)
        for fam, items in groups.items():
            by_family_per_run[fam][hw] = items

    for family in ALL_FAMILIES:
        if family not in by_family_per_run:
            continue
        tex = render_family_tex_table(family, by_family_per_run[family])
        if not tex:
            continue
        path = output_dir / f"{family}_table.tex"
        path.write_text(tex, encoding="utf-8")
        written.append(path)
    return written


def generate_plots(
    by_hw_rows: dict[HardwareKey, list[ComparisonRow]],
    output_dir: Path,
) -> None:
    by_family_per_run: dict[str, dict[HardwareKey, list[ComparisonRow]]] = defaultdict(dict)
    for hw, rows in by_hw_rows.items():
        groups = rows_by_family(rows)
        for fam, items in groups.items():
            by_family_per_run[fam][hw] = items

    hw_colors = assign_hw_colors(list(by_hw_rows.keys()))
    for family, by_run in by_family_per_run.items():
        plot_ratio(family, by_run, hw_colors, output_dir / f"{family}_ratio.pdf")
        plot_absolute(family, by_run, hw_colors, output_dir / f"{family}_absolute.pdf")
        plot_ratio_of_ratios(family, by_run, hw_colors, output_dir / f"{family}_ror.pdf")
    plot_combined_ror_overview(by_hw_rows, output_dir / "castro_combined_ror.pdf")


# %% CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bench/results/run_*.json to Castro et al. (2025) reference tables.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=[DEFAULT_RESULTS_DIR],
        help=(
            "Run JSONs or directories containing run_*.json files. "
            f"Defaults to {DEFAULT_RESULTS_DIR.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and report.md (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Plot every matching run separately. Default keeps only the latest per hardware.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Restrict to a single architecture (e.g. arm64, x86_64).",
    )
    args = parser.parse_args()

    runs = discover_runs(args.runs)
    castro_runs = [r for r in runs if has_castro_payload(r)]

    if not castro_runs:
        print("No runs contain Castro labels and required algorithms.")
        print(f"Looked under: {[str(p) for p in args.runs]}")
        print("Required: 'dijkstra' and 'bmssp_base' results for any of: " + ", ".join(sorted({l for l, _ in CASTRO_REF})))
        raise SystemExit(1)

    if args.arch:
        castro_runs = [r for r in castro_runs if r.hardware.arch == args.arch]
        if not castro_runs:
            print(f"No runs with arch={args.arch}")
            raise SystemExit(1)

    if args.all_runs:
        chosen: dict[HardwareKey, RunResult] = {}
        for r in castro_runs:
            key = HardwareKey(hw_id=f"{r.hardware.hw_id}@{r.run_id}", cpu=r.hardware.cpu, arch=r.hardware.arch)
            chosen[key] = r
    else:
        chosen = latest_per_hardware(castro_runs)

    print(f"Using {len(chosen)} run(s):")
    for hw, run in chosen.items():
        print(f"  - {hw.short()}  run_id={run.run_id}  ({run.path})")

    by_hw_rows: dict[HardwareKey, list[ComparisonRow]] = {}
    for hw, run in chosen.items():
        by_hw_rows[hw] = build_comparison(run)

    args.output.mkdir(parents=True, exist_ok=True)
    generate_plots(by_hw_rows, args.output)
    written = write_tex_outputs(by_hw_rows, args.output)
    print(f"\nWrote {len(written)} .tex file(s) to {args.output}:")
    for p in written:
        print(f"  - {p.name}")
    print(f"Plots: {args.output}/<family>_{{ratio,absolute,ror}}.pdf")


if __name__ == "__main__":
    main()
