"""
Generate README figures for GPTQ from scratch.
All data is hardcoded — no model loading required.

Usage: python generate_figures.py
Output: figures/*.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Colors
C_GREY = "#9e9e9e"
C_ORANGE = "#e67e22"
C_BLUE = "#2980b9"
C_RED = "#c0392b"
C_GREEN = "#27ae60"
C_FP16 = "#bdc3c7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# =========================================================================
# 1. Vanilla results — grouped bar chart (FP16 vs GPTQ 4-bit)
# =========================================================================
def fig_vanilla():
    models = ["GPT-2-XL\n(1.6B)", "OPT-1.3B\n(1.4B)", "Llama-3-8B\n(8B)"]
    fp16   = [14.79, 12.51, 5.49]
    gptq   = [15.58, 14.83, 7.73]

    x = np.arange(len(models))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, fp16, w, color=C_GREY, label="FP16 baseline", edgecolor="white")
    ax.bar(x + w/2, gptq, w, color=C_BLUE, label="GPTQ 4-bit", edgecolor="white")

    for i in range(len(models)):
        delta_pct = (gptq[i] - fp16[i]) / fp16[i] * 100
        ax.text(x[i] + w/2, gptq[i] + 0.15, f"+{delta_pct:.1f}%",
                ha="center", va="bottom", fontsize=10, color=C_BLUE, fontweight="bold")

    ax.set_ylabel("WikiText-2 Perplexity (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(gptq) * 1.2)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("GPTQ 4-bit Quantization — WikiText-2 Perplexity", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUT, "results_vanilla.png"))
    plt.close(fig)
    print("  -> results_vanilla.png")


# =========================================================================
# 2. Vanilla vs Optimized — grouped bar chart
# =========================================================================
def fig_vanilla_vs_optimized():
    models = ["GPT-2-XL\n(1.6B)", "OPT-1.3B\n(1.4B)", "Llama-3-8B\n(8B)"]
    fp16    = [14.79, 12.51, 5.49]
    vanilla = [15.58, 14.83, 7.73]
    optim   = [15.05, 12.90, 5.93]

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, fp16,    w, color=C_GREY,   label="FP16 baseline", edgecolor="white")
    b2 = ax.bar(x,     vanilla, w, color=C_ORANGE,  label="Vanilla GPTQ 4-bit", edgecolor="white")
    b3 = ax.bar(x + w, optim,   w, color=C_BLUE,    label="Optimized GPTQ 4-bit", edgecolor="white")

    # Annotate deltas
    for i in range(len(models)):
        delta_v = vanilla[i] - fp16[i]
        delta_o = optim[i] - fp16[i]
        ax.text(x[i],     vanilla[i] + 0.15, f"+{delta_v:.2f}", ha="center", va="bottom", fontsize=9, color=C_ORANGE, fontweight="bold")
        ax.text(x[i] + w, optim[i]   + 0.15, f"+{delta_o:.2f}", ha="center", va="bottom", fontsize=9, color=C_BLUE,   fontweight="bold")

    ax.set_ylabel("WikiText-2 Perplexity (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(vanilla) + 2.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("GPTQ 4-bit: Vanilla vs Optimized (g=128 + act-order + true-seq)", fontsize=13, fontweight="bold")

    fig.savefig(os.path.join(OUT, "results_vanilla_vs_optimized.png"))
    plt.close(fig)
    print("  -> results_vanilla_vs_optimized.png")


# =========================================================================
# 2. GPTQ vs RTN — log scale bar chart
# =========================================================================
def fig_gptq_vs_rtn():
    labels = ["FP16\nbaseline", "GPTQ\n4-bit", "Naive RTN\n4-bit"]
    values = [25.17, 35.45, 14434]
    colors = [C_GREY, C_BLUE, C_RED]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)

    ax.set_yscale("log")
    ax.set_ylabel("WikiText-2 Perplexity (log scale)")
    ax.set_title("Why GPTQ Matters: 400x Better Than Naive RTN", fontsize=13, fontweight="bold")

    # Annotate values
    for bar, val in zip(bars, values):
        y = bar.get_height()
        fmt = f"{val:,.0f}" if val > 100 else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, y * 1.15, fmt,
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add 400x annotation
    ax.annotate("400x", xy=(2, 14434), xytext=(1.5, 3000),
                fontsize=14, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=2))

    ax.set_ylim(10, 50000)

    fig.savefig(os.path.join(OUT, "gptq_vs_rtn.png"))
    plt.close(fig)
    print("  -> gptq_vs_rtn.png")


# =========================================================================
# 3. Extreme quantization — line plot
# =========================================================================
def fig_extreme_quantization():
    bits = [4, 3, 2]
    gptq = [35.45, 765.03, 11749.80]
    rtn  = [14434, 8559.39, 51045.48]
    fp16 = 25.17

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bits, gptq, "o-", color=C_BLUE, linewidth=2.5, markersize=9, label="GPTQ", zorder=3)
    ax.plot(bits, rtn,  "s-", color=C_RED,  linewidth=2.5, markersize=9, label="Naive RTN", zorder=3)
    ax.axhline(y=fp16, color=C_GREY, linestyle="--", linewidth=1.5, label=f"FP16 baseline ({fp16})", zorder=2)

    ax.set_yscale("log")
    ax.set_xlabel("Bit-Width")
    ax.set_ylabel("WikiText-2 Perplexity (log scale)")
    ax.set_xticks(bits)
    ax.set_xticklabels(["4-bit", "3-bit", "2-bit"])
    ax.invert_xaxis()
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("GPT-2 Small: Perplexity vs Bit-Width", fontsize=13, fontweight="bold")

    # Annotate key points
    for b, v in zip(bits, gptq):
        ax.annotate(f"{v:.0f}" if v > 100 else f"{v:.1f}",
                    xy=(b, v), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color=C_BLUE, fontweight="bold")

    fig.savefig(os.path.join(OUT, "extreme_quantization.png"))
    plt.close(fig)
    print("  -> extreme_quantization.png")


# =========================================================================
# 4. Algorithm diagram — flow chart
# =========================================================================
def fig_algorithm_diagram():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    boxes = [
        (0.3,  "Calibration Data\n(128 samples)"),
        (2.3,  "Hessian\n$H = X^\\top X / n$"),
        (4.3,  "Cholesky\n$H^{-1}$"),
        (6.3,  "Column-by-column\nquantize + compensate"),
        (8.5,  "Quantized\nWeights"),
    ]

    box_w = 1.7
    box_h = 1.2
    box_y = 0.4
    box_colors = [C_GREY, C_ORANGE, C_ORANGE, C_BLUE, C_GREEN]

    for i, (bx, text) in enumerate(boxes):
        rect = mpatches.FancyBboxPatch(
            (bx, box_y), box_w, box_h,
            boxstyle="round,pad=0.12",
            facecolor=box_colors[i], alpha=0.18,
            edgecolor=box_colors[i], linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(bx + box_w / 2, box_y + box_h / 2, text,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#2c3e50")

        # Arrow to next box
        if i < len(boxes) - 1:
            next_bx = boxes[i + 1][0]
            ax.annotate("", xy=(next_bx - 0.05, box_y + box_h / 2),
                        xytext=(bx + box_w + 0.05, box_y + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=2))

    ax.set_title("GPTQ Algorithm Pipeline", fontsize=13, fontweight="bold", pad=15)

    fig.savefig(os.path.join(OUT, "algorithm_diagram.png"))
    plt.close(fig)
    print("  -> algorithm_diagram.png")


# =========================================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig_vanilla()
    fig_vanilla_vs_optimized()
    fig_gptq_vs_rtn()
    fig_extreme_quantization()
    fig_algorithm_diagram()
    print(f"Done. All figures saved to {OUT}/")
