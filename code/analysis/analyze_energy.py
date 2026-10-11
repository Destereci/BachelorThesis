from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = r"D:\BachelorThesisGit\BachelorThesis\code\results\analysis\summary_all.csv"
OUT_DIR = Path("plots")
QUANT_ORDER = ["fp16", "int8", "int4"]

# Energy columns: lower = greener
ENERGY_COLS = ["j_per_output_token", "j_per_input_token", "j_per_sample", "total_joules"]
INFO_COLS = ["mean_input_tokens", "mean_output_tokens"]
PROPORTION_TASKS = {"code", "json"}

OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df["quantization"] = pd.Categorical(df["quantization"], QUANT_ORDER, ordered=True)

df["j_per_sample"] = df["total_joules"] / df["n_samples"]
df["j_per_correct"] = np.where(
    df["quality_primary"] > 0,
    df["total_joules"] / (df["quality_primary"] * df["n_samples"]),
    np.nan,
)

LABELS = {
    "j_per_output_token": "J/out tok",
    "j_per_input_token": "J/in tok",
    "j_per_sample": "J/sample",
    "total_joules": "total J",
    "j_per_correct": "J/correct",
    "mean_input_tokens": "mean in tok",
    "mean_output_tokens": "mean out tok",
}


def fmt(col: str, v: float) -> str:
    if np.isnan(v):
        return "–"
    if col == "total_joules":
        return f"{v:,.0f}"
    if col.startswith("mean_"):
        return f"{v:.1f}"
    return f"{v:.2f}"


def draw_table(sub: pd.DataFrame, task: str) -> None:
    sub = sub.sort_values(["family", "size_b", "quantization"])

    cols = INFO_COLS + ENERGY_COLS
    if task in PROPORTION_TASKS:
        cols.append("j_per_correct")
    colored = set(ENERGY_COLS) | {"j_per_correct"}

    row_labels = [f"{r.family} {r.size_b}B {r.quantization}" for r in sub.itertuples()]
    values = sub[cols].to_numpy(dtype=float)

    text = [[fmt(c, v) for c, v in zip(cols, row)] + [str(int(n))]
            for row, n in zip(values, sub["n_samples"])]

    cmap = plt.get_cmap("RdYlGn")
    norm = np.full_like(values, np.nan)
    for j, c in enumerate(cols):
        if c not in colored:
            continue
        col = values[:, j]
        if np.isnan(col).all():
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        n = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        norm[:, j] = 1 - n

    colours = []
    for i in range(len(sub)):
        row = [cmap(norm[i, j]) if not np.isnan(norm[i, j]) else "white"
               for j in range(len(cols))]
        colours.append(row + ["white"])

    col_labels = [LABELS[c] for c in cols] + ["n"]

    fig, ax = plt.subplots(figsize=(1.4 * len(col_labels) + 3, 0.45 * len(sub) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=text, rowLabels=row_labels, colLabels=col_labels,
                   cellColours=colours, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(f"Energy metrics – {task}", fontweight="bold", pad=12)

    out = OUT_DIR / f"energy_{task}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


for task, sub in df.groupby("task_type"):
    draw_table(sub, task)