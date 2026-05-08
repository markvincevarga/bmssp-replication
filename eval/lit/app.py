import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import re
import benchmark_analysis as ba

CRITERION_DIR = Path(__file__).resolve().parent.parent.parent / "target" / "criterion"
ba.CRITERION_DIR = CRITERION_DIR


def _algo_sort_key(name):
    if name == "dijkstra":
        return ("0_dijkstra", 0)
    if name == "delta_stepping":
        return ("0_delta_stepping", 0)
    m = re.match(r"(.+?)(\d+)$", name)
    if m:
        return (m.group(1), int(m.group(2)))
    return (name, -1)


def _edge_factor_sort_key(ef):
    m = re.match(r"(\d+)", ef)
    return int(m.group(1)) if m else 0


@st.cache_data
def discover():
    algos = sorted(ba.discover_algorithms(CRITERION_DIR), key=_algo_sort_key)
    all_nodes = set()
    all_edges = set()
    for algo in algos:
        nodes, edges = ba.discover_benchmark_configs(CRITERION_DIR, algo)
        all_nodes.update(nodes)
        all_edges.update(edges)
    return algos, sorted(all_nodes), sorted(all_edges, key=_edge_factor_sort_key)


@st.cache_data
def load_data(algorithms, node_counts, edge_factors):
    return ba.load_all_benchmarks(
        list(algorithms), list(node_counts), list(edge_factors)
    )


def build_chart(data, edge_factor, node_counts, algorithms, colors, y_scale):
    all_means = []
    for algo in algorithms:
        if algo in data:
            all_means.extend([d.mean for d in data[algo].values()])
    if not all_means:
        return None

    max_mean = max(all_means)
    scale, unit = ba.format_time_axis(max_mean)

    fig = go.Figure()
    for algo in algorithms:
        if algo not in data:
            continue
        algo_data = data[algo]
        nodes_with_data = [n for n in node_counts if n in algo_data]
        means = [algo_data[n].mean / scale for n in nodes_with_data]
        ci_upper = [(algo_data[n].ci_upper - algo_data[n].mean) / scale for n in nodes_with_data]
        ci_lower = [(algo_data[n].mean - algo_data[n].ci_lower) / scale for n in nodes_with_data]

        fig.add_trace(go.Scatter(
            name=algo.upper(),
            x=nodes_with_data,
            y=means,
            mode="lines+markers",
            marker_color=colors.get(algo, "#333"),
            error_y=dict(type="data", symmetric=False, array=ci_upper, arrayminus=ci_lower),
            hovertemplate="<b>%{x} nodes</b><br>%{y:.3f} " + unit + "<extra>" + algo.upper() + "</extra>",
        ))

    fig.update_layout(
        xaxis_title="Number of Nodes",
        yaxis_title=f"Execution Time ({unit})",
        xaxis_type="log",
        yaxis_type="log" if y_scale == "Logarithmic" else "linear",
        legend_title="Algorithm",
        height=500,
        margin=dict(t=40),
    )
    return fig


def build_ratio_chart(data, node_counts, algorithms, baseline, colors):
    if baseline not in data:
        return None

    baseline_data = data[baseline]
    nodes_with_baseline = [n for n in node_counts if n in baseline_data]
    if not nodes_with_baseline:
        return None

    fig = go.Figure()

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)

    for algo in algorithms:
        if algo not in data or algo == baseline:
            continue
        algo_data = data[algo]
        nodes = [n for n in nodes_with_baseline if n in algo_data]
        ratios = [algo_data[n].mean / baseline_data[n].mean for n in nodes]
        err_upper = [algo_data[n].ci_upper / baseline_data[n].mean - algo_data[n].mean / baseline_data[n].mean for n in nodes]
        err_lower = [algo_data[n].mean / baseline_data[n].mean - algo_data[n].ci_lower / baseline_data[n].mean for n in nodes]

        fig.add_trace(go.Scatter(
            name=algo.upper(),
            x=nodes,
            y=ratios,
            mode="lines+markers",
            marker_color=colors.get(algo, "#333"),
            error_y=dict(type="data", symmetric=False, array=err_upper, arrayminus=err_lower),
            hovertemplate="<b>%{x} nodes</b><br>%{y:.3f}x<extra>" + algo.upper() + "</extra>",
        ))

    fig.update_layout(
        xaxis_title="Number of Nodes",
        yaxis_title=f"Time Ratio (vs {baseline.upper()})",
        xaxis_type="log",
        legend_title="Algorithm",
        height=500,
        margin=dict(t=40),
    )
    return fig


def build_table(data, edge_factor, node_counts, algorithms, baseline):
    if edge_factor not in data:
        return None
    ef_data = data[edge_factor]
    rows = []
    for nodes in node_counts:
        row = {"Nodes": nodes}
        baseline_mean = None
        if baseline in ef_data and nodes in ef_data[baseline]:
            baseline_mean = ef_data[baseline][nodes].mean

        for algo in algorithms:
            if algo in ef_data and nodes in ef_data[algo]:
                mean = ef_data[algo][nodes].mean
                row[algo.upper()] = ba.format_time(mean)
                if baseline_mean and algo != baseline:
                    row[f"{algo.upper()} ratio"] = f"{mean / baseline_mean:.3f}x"
            else:
                row[algo.upper()] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise(data, edge_factor, node_counts, algorithms):
    if edge_factor not in data:
        return None
    ef_data = data[edge_factor]
    bmssp = [a for a in algorithms if a.startswith("bmssp")]
    if len(bmssp) < 2:
        return None

    sections = []
    for i in range(len(bmssp) - 1):
        base, compare = bmssp[i], bmssp[i + 1]
        rows = []
        for nodes in node_counts:
            b = ef_data.get(base, {}).get(nodes)
            c = ef_data.get(compare, {}).get(nodes)
            if b and c:
                speedup = b.mean / c.mean
                pct = (1 - c.mean / b.mean) * 100
                rows.append({
                    "Nodes": nodes,
                    base.upper(): ba.format_time(b.mean),
                    compare.upper(): ba.format_time(c.mean),
                    "Speedup": f"{speedup:.2f}x",
                    "Change": f"{pct:+.1f}%",
                })
        if rows:
            sections.append((f"{base.upper()} -> {compare.upper()}", pd.DataFrame(rows)))
    return sections


st.set_page_config(page_title="Benchmark Explorer", layout="wide")
st.title("Benchmark Explorer")

algorithms, node_counts, edge_factors = discover()

if not algorithms:
    st.error("No benchmark data found. Run `cargo bench` first.")
    st.stop()

selected_algos = st.sidebar.multiselect("Algorithms", algorithms, default=algorithms)
selected_edges = st.sidebar.multiselect("Edge Factors", edge_factors, default=edge_factors)

if len(node_counts) >= 2:
    node_range = st.sidebar.select_slider(
        "Node Range",
        options=node_counts,
        value=(node_counts[0], node_counts[-1]),
    )
    filtered_nodes = [n for n in node_counts if node_range[0] <= n <= node_range[1]]
else:
    filtered_nodes = node_counts

baseline_default = selected_algos.index("dijkstra") if "dijkstra" in selected_algos else 0
baseline = st.sidebar.selectbox(
    "Ratio Baseline",
    selected_algos,
    index=baseline_default,
) if selected_algos else None

y_scale = st.sidebar.radio("Y-Axis Scale", ["Logarithmic", "Linear"])

if not selected_algos:
    st.info("Select at least one algorithm.")
    st.stop()

if not selected_edges:
    st.info("Select at least one edge factor.")
    st.stop()

all_data = load_data(
    tuple(sorted(selected_algos, key=_algo_sort_key)),
    tuple(sorted(filtered_nodes)),
    tuple(sorted(selected_edges, key=_edge_factor_sort_key)),
)

colors = ba.assign_colors(algorithms)

tabs = st.tabs([f"Edge Factor {ef}" for ef in selected_edges])
for tab, ef in zip(tabs, selected_edges):
    with tab:
        if ef in all_data:
            fig = build_chart(all_data[ef], ef, filtered_nodes, selected_algos, colors, y_scale)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data for edge factor {ef}")

            if baseline:
                ratio_fig = build_ratio_chart(all_data[ef], filtered_nodes, selected_algos, baseline, colors)
                if ratio_fig:
                    st.plotly_chart(ratio_fig, use_container_width=True)

            df = build_table(all_data, ef, filtered_nodes, selected_algos, baseline)
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Pairwise Speedups"):
    for ef in selected_edges:
        sections = build_pairwise(all_data, ef, filtered_nodes, selected_algos)
        if sections:
            st.subheader(f"Edge Factor {ef}")
            for title, df in sections:
                st.caption(title)
                st.dataframe(df, use_container_width=True, hide_index=True)
