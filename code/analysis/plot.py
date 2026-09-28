from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# Visual style 


MODEL_COLOURS: dict[str, str] = {
    "phi3":     "#4C72B0",   # blue
    "mistral":  "#DD8452",   # orange
    "llama3":   "#55A868",   # green
    "tinyllama":"#C44E52",   # red
    "qwen":     "#8172B2",   # purple
}


QUANT_MARKERS: dict[str, str] = {
    "int8": "o",
    "int4": "^",
}

QUANT_LABELS: dict[str, str] = {
    "int8": "INT8",
    "int4": "INT4",
}


# Plot

def plot_task(df_task: pd.DataFrame, task_type: str, save_path: Path | None) -> None:
    """
    One plot per task type.
    Each point = one (model, quantization) configuration.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    
    # Convert pct columns to fractions e.g. 32.1 → 0.321
    all_vals = pd.concat([
        df_task["delta_energy_pct"].dropna() / 100,
        df_task["delta_quality_pct"].dropna() / 100,
    ])
    pad = 0.08
    lo = min(all_vals.min(), 0) - pad
    hi = max(all_vals.max(), 0) + pad

    # Make axes square
    lim = (min(lo, -0.05), max(hi, 0.05))
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # Diagonal: EQ = 0
    diag = np.linspace(lim[0], lim[1], 200)
    ax.plot(diag, diag,
            color="black", linewidth=1.2, linestyle="--",
            label=r"$\Delta E = \Delta Q$  (EQ = 0)", zorder=2)

    ax.fill_between(diag, lim[0], diag,
                    alpha=0.06, color="green",
                    label="EQ > 0  (energy savings dominate)")
    ax.fill_between(diag, diag, lim[1],
                    alpha=0.06, color="red",
                    label="EQ < 0  (quality loss dominates)")

    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")

    # Data points
    plotted_families = set()

    for _, row in df_task.iterrows():
        de = row["delta_energy_pct"] / 100
        dq = row["delta_quality_pct"] / 100
        family = str(row["family"]).lower()
        quant  = str(row["quantization"]).lower()

        if pd.isna(de) or pd.isna(dq):
            continue

        color  = MODEL_COLOURS.get(family, "#333333")
        marker = QUANT_MARKERS.get(quant, "s")

        ax.scatter(de, dq,
                   color=color,
                   marker=marker,
                   s=90,
                   zorder=5,
                   edgecolors="white",
                   linewidths=0.6)

        # Label each point
        label = f"{row['family']} {QUANT_LABELS.get(quant, quant.upper())}"
        ax.annotate(
            label,
            xy=(de, dq),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
        )

        plotted_families.add(family)

    # Legend 
    colour_handles = [
        mpatches.Patch(color=MODEL_COLOURS.get(f, "#333333"), label=f.capitalize())
        for f in sorted(plotted_families)
    ]

    marker_handles = [
        plt.Line2D([0], [0],
                   marker=QUANT_MARKERS[q],
                   color="grey",
                   linestyle="None",
                   markersize=7,
                   label=QUANT_LABELS[q])
        for q in QUANT_MARKERS
    ]

    region_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                   label=r"$\Delta E = \Delta Q$  (EQ = 0)"),
        mpatches.Patch(color="green", alpha=0.15, label="EQ > 0"),
        mpatches.Patch(color="red",   alpha=0.15, label="EQ < 0"),
    ]

    legend = ax.legend(
        handles=colour_handles + marker_handles + region_handles,
        fontsize=8,
        loc="upper left",
        framealpha=0.9,
        title="Model / Quantization",
        title_fontsize=8,
    )

    # Axes labels and title 
    ax.set_xlabel(r"Relative energy savings $\Delta E$  (higher → more efficient)",
                  fontsize=10)
    ax.set_ylabel(r"Relative quality degradation $\Delta Q$  (lower → less degraded)",
                  fontsize=10)
    ax.set_title(f"Energy–Quality Trade-off  |  Task: {task_type}",
                 fontsize=11, fontweight="bold")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    # Quadrant annotations
    ax.text(lim[1] - pad * 0.5, lim[0] + pad * 0.3,
            "Desirable\n(high savings, low loss)",
            ha="right", va="bottom", fontsize=7.5,
            color="darkgreen", style="italic")

    ax.text(lim[0] + pad * 0.3, lim[1] - pad * 0.3,
            "Undesirable\n(low savings, high loss)",
            ha="left", va="top", fontsize=7.5,
            color="darkred", style="italic")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot EQ trade-off from eq_summary.csv")
    p.add_argument("--eq-csv",   type=Path, required=True,
                   help="Path to eq_summary.csv produced by analyze_eq.py")
    p.add_argument("--task",     default="all",
                   help="Task type to plot, or 'all' for one plot per task")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Directory to save plots (PNG). If omitted, shows interactively.")
    args = p.parse_args()

    df = pd.read_csv(args.eq_csv)

    if df.empty:
        print("No data found in CSV.")
        return

    tasks = df["task_type"].unique() if args.task == "all" else [args.task]

    for task in tasks:
        df_task = df[df["task_type"] == task].copy()
        if df_task.empty:
            print(f"No data for task: {task}")
            continue

        save_path = None
        if args.save_dir:
            save_path = args.save_dir / f"eq_tradeoff__{task}.png"

        plot_task(df_task, task, save_path)


if __name__ == "__main__":
    main()