from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = "results/analysis/summary_all.csv"   # written by analyze.py
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

QUANT_ORDER = ["fp16", "int8", "int4"]
TASKS = ["code", "json", "summarization"]
MARKERS = {"fp16": "o", "int8": "s", "int4": "^"}
ENERGY_COL = "j_per_sample"


df = pd.read_csv(CSV_PATH)
df["quantization"] = pd.Categorical(df["quantization"], QUANT_ORDER, ordered=True)
df["model_label"] = df["model"].str.replace(r"_(fp16|int8|int4)$", "", regex=True)
df["j_per_sample"] = df["total_joules"] / df["n_samples"]
if "quality_empty" not in df:
    df["quality_empty"] = np.nan


def is_broken(r) -> bool:
    q, t = r.quantization, r.task_type
    return (
        (r.family == "qwen2.5_14b" and q == "int4" and (t == "json" or t == "code"))
        or (r.family == "phi3" and q == "int4" and t == "json")
    )


df["broken"] = df.apply(is_broken, axis=1)

models = sorted(df["model_label"].unique())
colors = dict(zip(models, plt.cm.tab10(np.arange(len(models)))))


def legend_handles():
    h = [plt.Line2D([], [], marker="o", ls="", color=colors[m], label=m) for m in models]
    h += [plt.Line2D([], [], marker=MARKERS[q], ls="", color="gray", label=q) for q in QUANT_ORDER]
    h.append(plt.Line2D([], [], marker="X", ls="", mfc="none", color="gray", label="broken cell"))
    return h


# ---------- Plot 1: energy vs quality, one panel per task ----------

def panel_energy_quality(ax, task):
    sub = df[df.task_type == task]
    for r in sub.itertuples():
        x = getattr(r, ENERGY_COL)
        if r.broken:
            ax.scatter(x, r.quality_primary, marker="X", s=90,
                       facecolors="none", edgecolors="gray", linewidths=1.5)
        else:
            size = 60 + 500 * (0 if np.isnan(r.quality_empty) else r.quality_empty)
            ax.scatter(x, r.quality_primary, marker=MARKERS[r.quantization],
                       s=size, color=colors[r.model_label], alpha=0.85,
                       edgecolors="black", linewidths=0.6)

    # Pareto frontier: minimal energy, maximal quality (non-broken only)
    ok = sub[~sub.broken].sort_values(ENERGY_COL)
    best, fx, fy = -np.inf, [], []
    for r in ok.itertuples():
        if r.quality_primary > best:
            best = r.quality_primary
            fx.append(getattr(r, ENERGY_COL))
            fy.append(r.quality_primary)
    ax.step(fx, fy, where="post", color="black", ls="--", lw=1)

    ax.set_title(f"Energy per request vs. quality: {task}", fontweight="bold")
    ax.set_xlabel("Mean energy per request [J] (lower = better)")
    ax.set_ylabel("quality_primary  (higher = better)")
    ax.grid(alpha=0.3)


for task in TASKS:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    panel_energy_quality(ax, task)
    handles = legend_handles() + [
        plt.Line2D([], [], color="black", ls="--", lw=1, label="Pareto frontier")
    ]
    ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT_DIR / f"1_{task}_energy_per_token_vs_quality.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)




# ---------- Plot 2: total tokens vs J/sample, J/token as slope ----------
df["mean_processed_tokens"] = (
    df["mean_input_tokens"] + df["mean_output_tokens"]
)
fig, ax = plt.subplots(figsize=(9, 6))
task_colors = dict(zip(TASKS, ["tab:blue", "tab:orange", "tab:green"]))
for r in df.itertuples():
    if r.broken:
        ax.scatter(r.total_processed_tokens, r.j_per_sample, marker="X", s=90,
                   facecolors="none", edgecolors="gray", linewidths=1.5)
    else:
        ax.scatter(r.total_processed_tokens, r.j_per_sample, marker=MARKERS[r.quantization],
                   s=70, color=task_colors[r.task_type], edgecolors="black", linewidths=0.6)

lo, hi = df.total_processed_tokens.min() * 0.8, df.total_processed_tokens.max() * 1.25
ylo, yhi = df.j_per_sample.min() * 0.7, df.j_per_sample.max() * 1.4
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(ylo, yhi)

# iso-lines: J per token (prefill + decode) = c
for c in (0.25, 0.5, 1, 2.5, 5, 10):
    xs = np.array([lo, hi])
    ax.plot(xs, c * xs, color="gray", ls=":", lw=1)
    x_lab = min(hi, yhi / c) * 0.97          # where the line leaves the plot
    if x_lab * c > ylo and x_lab > lo:
        exits_top = yhi / c < hi
        ax.text(x_lab, c * x_lab, f"{c} J/processed token", fontsize=8, color="gray",
                ha="right", va="top" if exits_top else "bottom")

ax.set_xticks([300, 500, 800])
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Mean processed tokens per request (input + output)")
ax.set_ylabel("Energy per sample [J]")
ax.set_title("Request energy vs. token volume (dotted lines: J/processed token)")
h = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in task_colors.items()]
h += [plt.Line2D([], [], marker=MARKERS[q], ls="", color="gray", label=q) for q in QUANT_ORDER]
h.append(plt.Line2D([], [], marker="X", ls="", mfc="none", color="gray", label="broken cell"))
ax.legend(handles=h, loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.savefig(OUT_DIR / "2_total_processed_tokens_vs_energy.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Plot 3: change relative to fp16 ----------
base = df[df.quantization == "fp16"].set_index(["model_label", "task_type"])

def vs_base(r, col):
    return base.loc[(r.model_label, r.task_type), col]

df["speed_change_pct"] = [(r.tok_per_s / vs_base(r, "tok_per_s") - 1) * 100 for r in df.itertuples()]
df["energy_change_pct"] = [(r.j_per_sample / vs_base(r, "j_per_sample") - 1) * 100 for r in df.itertuples()]
df["quality_delta"] = [r.quality_primary - vs_base(r, "quality_primary") for r in df.itertuples()]

metrics = [
    ("speed_change_pct", "Speed vs. fp16 [%]\n(above 0 = faster)"),
    ("energy_change_pct", "Energy vs. fp16 [%]\n(below 0 = saves energy)"),
    ("quality_delta", "Quality vs. fp16 [points]\n(above 0 = better)"),
]
fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex="col")
qcolors = {"int8": "tab:blue", "int4": "tab:red"}
w = 0.38
for j, task in enumerate(TASKS):
    sub = df[(df.task_type == task) & (df.quantization != "fp16")]
    for i, (col, ylabel) in enumerate(metrics):
        ax = axes[i, j]
        for k, q in enumerate(["int8", "int4"]):
            for x, m in enumerate(models):
                row = sub[(sub.model_label == m) & (sub.quantization == q)]
                if row.empty:
                    continue
                r = row.iloc[0]
                ax.bar(x + (k - 0.5) * w, r[col], w,
                       color="lightgray" if r.broken else qcolors[q],
                       hatch="//" if r.broken else None, edgecolor="black",
                       linewidth=0.5, label=q if x == 0 else None)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.set_ylabel(ylabel)
        if i == 0:
            ax.set_title(task, fontweight="bold")
axes[0, 0].legend()
fig.suptitle("Quantization effect relative to fp16 (hatched gray = broken cell)")
fig.tight_layout()
fig.savefig(OUT_DIR / "3_change_vs_fp16.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------- Plot 4: tok/s vs J/token, power as iso-lines ----------
df["j_per_token_sum"] = df["total_joules"] / (df["n_samples"] * df["mean_output_tokens"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, task in zip(axes, TASKS):
    sub = df[df.task_type == task]
    for r in sub.itertuples():
        if r.broken:
            ax.scatter(r.tok_per_s, r.j_per_token_sum, marker="X", s=90,
                       facecolors="none", edgecolors="gray", linewidths=1.5)
        else:
            ax.scatter(r.tok_per_s, r.j_per_token_sum, marker=MARKERS[r.quantization],
                       s=80, color=colors[r.model_label], edgecolors="black", linewidths=0.6)
    xs = np.geomspace(sub.tok_per_s.min() * 0.7, sub.tok_per_s.max() * 1.4, 50)
    for watts in (100, 150, 200, 250):
        ax.plot(xs, watts / xs, color="gray", ls=":", lw=1)
        ax.text(xs[-1], watts / xs[-1], f"{watts} W", fontsize=8, color="gray", ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(sub.j_per_token_sum.min() * 0.7, sub.j_per_token_sum.max() * 1.4)
    ax.set_title(task, fontweight="bold")
    ax.set_xlabel("Decode throughput [tok/s]  (right = faster)")
    ax.set_ylabel("Energy per output token [J]  (down = better)")
    ax.grid(alpha=0.3, which="both")
axes[-1].legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Speed vs. energy per token (dotted lines = constant average power)")
fig.savefig(OUT_DIR / "4_speed_vs_energy.png", dpi=200, bbox_inches="tight")
plt.close(fig)




# ---------- Plot 5: measurement sanity check (energy = power x time) ----------
# Left:  end-to-end throughput vs. model size, with the memory-bandwidth limit
#        for batch size 1 (every decoded token reads all weights once).
# Right: energy per request vs. latency; dotted lines = constant average power.
BANDWIDTH = 672e9                                   # RTX 8000, bytes/s
BYTES_PER_PARAM = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
task_colors = dict(zip(TASKS, ["tab:blue", "tab:orange", "tab:green"]))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- (a) throughput vs. model size (decode-dominated tasks only) ---
sizes = np.geomspace(0.4, 17, 100)
for q in QUANT_ORDER:
    axA.plot(sizes, BANDWIDTH / (sizes * 1e9 * BYTES_PER_PARAM[q]),
             color="gray", ls=["-", "--", ":"][QUANT_ORDER.index(q)], lw=1,
             label=f"bandwidth limit ({q})")
for r in df[df.task_type.isin(["code", "json"])].itertuples():
    if r.broken:
        axA.scatter(r.size_b, r.tok_per_s, marker="X", s=90,
                    facecolors="none", edgecolors="gray", linewidths=1.5)
    else:
        axA.scatter(r.size_b, r.tok_per_s, marker=MARKERS[r.quantization], s=70,
                    color=task_colors[r.task_type], edgecolors="black", linewidths=0.6)
axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlim(0.4, 17)
axA.set_ylim(8, 4000)
axA.set_xticks([0.5, 3.8, 7, 14])
axA.set_xticklabels(["0.5", "3.8", "7", "14"])
axA.xaxis.set_minor_formatter(plt.NullFormatter())
axA.set_xlabel("Model size [B parameters]")
axA.set_ylabel("End-to-end throughput [output tokens/s]")
axA.set_title("(a) Throughput vs. model size (code, JSON)", fontweight="bold")
axA.grid(alpha=0.3, which="both")

# --- (b) energy vs. latency, constant-power lines ---
lo, hi = df.mean_latency_s.min() * 0.7, df.mean_latency_s.max() * 1.4
ylo, yhi = df.j_per_sample.min() * 0.7, df.j_per_sample.max() * 1.4
ts = np.array([lo, hi])
for watts in (100, 150, 200):
    axB.plot(ts, watts * ts, color="gray", ls=":", lw=1)
    x_lab = min(hi, yhi / watts) * 0.97             # where the line leaves the plot
    axB.text(x_lab, watts * x_lab, f"{watts} W", fontsize=8, color="gray", ha="right", va="top")
for r in df.itertuples():
    if r.broken:
        axB.scatter(r.mean_latency_s, r.j_per_sample, marker="X", s=90,
                    facecolors="none", edgecolors="gray", linewidths=1.5)
    else:
        axB.scatter(r.mean_latency_s, r.j_per_sample, marker=MARKERS[r.quantization], s=70,
                    color=task_colors[r.task_type], edgecolors="black", linewidths=0.6)
axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlim(lo, hi)
axB.set_ylim(ylo, yhi)
axB.set_xlabel("Mean latency per request [s]")
axB.set_ylabel("Mean energy per request [J]")
axB.set_title("(b) Energy vs. latency (dotted: constant power)", fontweight="bold")
axB.grid(alpha=0.3, which="both")

h = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in task_colors.items()]
h += [plt.Line2D([], [], marker=MARKERS[q], ls="", color="gray", label=q) for q in QUANT_ORDER]
h.append(plt.Line2D([], [], marker="X", ls="", mfc="none", color="gray", label="broken cell"))
axB.legend(handles=h, loc="upper left", fontsize=8)
axA.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "5_measurement_sanity_check.png", dpi=200, bbox_inches="tight")
plt.close(fig)







cmp = df[(df.quantization == "int4") & ~df.broken][
    ["model_label", "task_type", "speed_change_pct", "energy_change_pct", "quality_delta"]
].round(1)
print(cmp.to_string(index=False))
n = len(cmp)
print(f"\nint4 faster than fp16:        {(cmp.speed_change_pct > 0).sum()}/{n}")
print(f"int4 uses less energy/sample: {(cmp.energy_change_pct < 0).sum()}/{n}")
print(f"int4 loses quality:           {(cmp.quality_delta < 0).sum()}/{n}")

# is fp16 the most correct *within each task*?
best = df[~df.broken].loc[df[~df.broken].groupby("task_type").quality_primary.idxmax()]
print(best[["task_type", "model", "quality_primary"]].to_string(index=False))


print("saved 3 plots to", OUT_DIR)

