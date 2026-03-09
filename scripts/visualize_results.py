#!/usr/bin/env python3
"""
visualize_results.py - 可视化联邦学习公平性实验结果
用法:
    conda run -n PFL-Fair python3 visualize_results.py
    conda run -n PFL-Fair python3 visualize_results.py --results-dir results --output-dir results/figures --eval-gap 5
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── 算法显示顺序与颜色 ────────────────────────────────────────────────────────
ALGO_ORDER = ["FedAvg_Fair", "FairFed", "FedALA_Fair", "PFL-Fair"]
ALGO_COLORS = {
    "FedAvg_Fair": "#1f77b4",  # 蓝
    "FairFed": "#ff7f0e",  # 橙
    "FedALA_Fair": "#2ca02c",  # 绿
    "PFL-Fair": "#d62728",  # 红
}
ALGO_MARKERS = {
    "FedAvg_Fair": "o",
    "FairFed": "s",
    "FedALA_Fair": "^",
    "PFL-Fair": "D",
}

# ── 指标配置 ──────────────────────────────────────────────────────────────────
METRICS_CONFIG = {
    "rs_test_acc": {"label": "Test Accuracy", "direction": "higher"},
    "rs_train_loss": {"label": "Train Loss", "direction": "lower"},
    "rs_eod": {"label": "EOD (Equalized Odds Diff)", "direction": "lower"},
    "rs_acc_gap": {"label": "AccGap (|g0−g1|)", "direction": "lower"},
    "rs_acc_std": {"label": "AccStd (clients)", "direction": "lower"},
    "rs_acc_worst": {"label": "AccWorst-10%", "direction": "higher"},
}

# 汇总表列定义
TABLE_COLS = [
    ("rs_test_acc", "Acc↑", "{:.4f}"),
    ("rs_eod", "EOD↓", "{:.4f}"),
    ("rs_acc_gap", "AccGap↓", "{:.4f}"),
    ("rs_acc_std", "AccStd↓", "{:.4f}"),
    ("rs_acc_worst", "AccW10↑", "{:.4f}"),
    ("rs_train_loss", "Loss↓", "{:.4f}"),
]


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def extract_algo_name(filename: str) -> str:
    """
    从文件名 '{Dataset}_{AlgoName}_{run_tag}_{seed}.h5' 中提取算法名称。
    规则：去掉 Dataset_ 前缀后，收集以大写字母或含 '-' 开头的 token 作为算法名。
    """
    stem = filename.replace(".h5", "")
    # 去掉第一个 '_' 之前的 dataset 前缀
    rest = stem.split("_", 1)[1] if "_" in stem else stem
    tokens = rest.split("_")
    parts = []
    for t in tokens:
        if t and (t[0].isupper() or "-" in t):
            parts.append(t)
        else:
            break
    return "_".join(parts) if parts else stem


def load_results(results_dir: str) -> dict:
    """加载目录下所有 .h5 文件，返回 {algo_name: {metric_key: np.array}}。"""
    data = {}
    for fpath in sorted(Path(results_dir).glob("*.h5")):
        algo = extract_algo_name(fpath.name)
        with h5py.File(fpath, "r") as f:
            metrics = {k: f[k][:] for k in f.keys() if len(f[k][:]) > 0}
        data[algo] = metrics
    return data


def ordered_algos(data: dict) -> list:
    """按预设顺序返回存在数据的算法列表。"""
    seen = [a for a in ALGO_ORDER if a in data]
    extra = [a for a in data if a not in seen]
    return seen + extra


# ── 汇总表 ────────────────────────────────────────────────────────────────────


def print_summary_table(data: dict):
    algos = ordered_algos(data)
    col_w = 12

    print("\n" + "=" * 90)
    print("  实验结果汇总表（最终轮次）")
    print("=" * 90)

    header = f"{'Algorithm':<18}" + "".join(f"{col[1]:>{col_w}}" for col in TABLE_COLS)
    print(header)
    print("-" * len(header))

    for algo in algos:
        row = f"{algo:<18}"
        for key, _, fmt in TABLE_COLS:
            m = data[algo]
            val = fmt.format(m[key][-1]) if key in m and len(m[key]) > 0 else "N/A"
            row += f"{val:>{col_w}}"
        print(row)

    print("=" * 90)
    print("  ↑: 越高越好   ↓: 越低越好")
    print("  EOD  = Equalized Odds Difference（公平性，越小越公平）")
    print("  AccGap = |Acc_group0 − Acc_group1|（组间精度差）")
    print("  AccStd = 客户端精度标准差（个性化程度）")
    print("  AccW10 = 最差10%客户端精度（最弱群体保障）")
    print()


# ── 训练曲线图 ────────────────────────────────────────────────────────────────


def plot_training_curves(data: dict, output_dir: str, eval_gap: int):
    os.makedirs(output_dir, exist_ok=True)
    algos = ordered_algos(data)

    all_metrics = [
        "rs_test_acc",
        "rs_train_loss",
        "rs_eod",
        "rs_acc_gap",
        "rs_acc_std",
        "rs_acc_worst",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        "Federated Learning Fairness Comparison — Adult Dataset",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for ax, metric_key in zip(axes.flat, all_metrics):
        cfg = METRICS_CONFIG[metric_key]
        arrow = "↑" if cfg["direction"] == "higher" else "↓"
        for algo in algos:
            if metric_key not in data[algo]:
                continue
            vals = data[algo][metric_key]
            rounds = np.arange(len(vals)) * eval_gap
            ax.plot(
                rounds,
                vals,
                marker=ALGO_MARKERS.get(algo, "o"),
                markersize=4,
                color=ALGO_COLORS.get(algo, None),
                label=algo,
                linewidth=1.8,
                markevery=2,
            )

        ax.set_title(f"{cfg['label']}  ({arrow} better)", fontsize=10)
        ax.set_xlabel("Communication Round", fontsize=9)
        ax.set_ylabel(cfg["label"], fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {out}")


# ── 精度-公平权衡散点图 ────────────────────────────────────────────────────────


def plot_tradeoff(data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    algos = ordered_algos(data)

    fairness_axes = [
        ("rs_eod", "EOD  (↓ better)"),
        ("rs_acc_gap", "AccGap  (↓ better)"),
        ("rs_acc_worst", "AccWorst-10%  (↑ better)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Accuracy–Fairness Trade-off (Final Round) — Adult Dataset",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (fm_key, fm_label) in zip(axes, fairness_axes):
        for algo in algos:
            d = data[algo]
            if "rs_test_acc" not in d or fm_key not in d:
                continue
            acc = d["rs_test_acc"][-1]
            fm = d[fm_key][-1]
            color = ALGO_COLORS.get(algo, "grey")
            marker = ALGO_MARKERS.get(algo, "o")
            ax.scatter(
                acc,
                fm,
                s=160,
                color=color,
                marker=marker,
                label=algo,
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
            )
            ax.annotate(
                algo,
                (acc, fm),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color=color,
                fontweight="bold",
            )

        ax.set_xlabel("Test Accuracy  (↑ better)", fontsize=10)
        ax.set_ylabel(fm_label, fontsize=10)
        ax.set_title(f"Acc vs {fm_label.split('(')[0].strip()}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)

    # 统一图例
    legend_patches = [
        mpatches.Patch(color=ALGO_COLORS.get(a, "grey"), label=a)
        for a in algos
        if a in data
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(algos),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    out = os.path.join(output_dir, "tradeoff_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {out}")


# ── 雷达图（最终轮次综合对比）────────────────────────────────────────────────


def plot_radar(data: dict, output_dir: str):
    """雷达图 - 对多指标综合可视化，数值越大越好（需对 lower-is-better 指标取反）。"""
    os.makedirs(output_dir, exist_ok=True)
    algos = ordered_algos(data)

    # 雷达图维度（全部归一化到 [0,1]，越大越好）
    radar_keys = [
        ("rs_test_acc", "Accuracy", "higher"),
        ("rs_acc_worst", "AccWorst", "higher"),
        ("rs_eod", "1−EOD", "lower"),  # invert
        ("rs_acc_gap", "1−AccGap", "lower"),  # invert
        ("rs_acc_std", "1−AccStd", "lower"),  # invert
    ]
    labels = [r[1] for r in radar_keys]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 收集各维度数值
    raw = {}
    for algo in algos:
        row = []
        for key, _, direction in radar_keys:
            d = data[algo]
            val = d[key][-1] if key in d and len(d[key]) > 0 else 0.0
            row.append((val, direction))
        raw[algo] = row

    # 各维度归一化（min-max across algos）
    normalized = {a: [] for a in algos}
    for i in range(num_vars):
        vals = [raw[a][i][0] for a in algos]
        direction = radar_keys[i][2]
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi != lo else 1.0
        for algo in algos:
            v = raw[algo][i][0]
            norm = (v - lo) / rng  # [0,1]
            if direction == "lower":
                norm = 1.0 - norm  # 低→好 → 取反
            normalized[algo].append(norm)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title(
        "Comprehensive Comparison (Final Round)\nRadar Chart — Adult Dataset",
        fontsize=11,
        fontweight="bold",
        pad=20,
    )

    for algo in algos:
        vals = normalized[algo] + normalized[algo][:1]
        color = ALGO_COLORS.get(algo, "grey")
        marker = ALGO_MARKERS.get(algo, "o")
        ax.plot(
            angles,
            vals,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            label=algo,
        )
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {out}")


# ── 主函数 ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="可视化 PFL 公平性实验结果")
    parser.add_argument("--results-dir", default="results", help="HDF5 文件所在目录")
    parser.add_argument("--output-dir", default="results/figures", help="图片输出目录")
    parser.add_argument("--eval-gap", type=int, default=5, help="评估间隔（轮次）")
    args = parser.parse_args()

    print(f"\n加载结果目录: {args.results_dir}")
    data = load_results(args.results_dir)
    if not data:
        print("错误：未找到任何 .h5 结果文件！")
        return

    print(f"发现算法: {list(data.keys())}")

    # ① 汇总表（终端打印）
    print_summary_table(data)

    # ② 训练曲线
    print("生成训练曲线图...")
    plot_training_curves(data, args.output_dir, args.eval_gap)

    # ③ 精度-公平散点图
    print("生成权衡散点图...")
    plot_tradeoff(data, args.output_dir)

    # ④ 雷达图
    print("生成雷达图...")
    plot_radar(data, args.output_dir)

    print(f"\n全部完成！图像已保存到: {args.output_dir}/")


if __name__ == "__main__":
    main()
