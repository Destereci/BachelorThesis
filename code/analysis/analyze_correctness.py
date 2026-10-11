from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = r"D:\BachelorThesisGit\BachelorThesis\code\results\analysis\summary_all.csv"
OUT_DIR = Path("plots")
QUANT_ORDER = ["fp16", "int8", "int4"]
LOWER_IS_BETTER = {"quality_empty", "quality_schema_error"}

OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

df["quantization"] = pd.Categorical(df["quantization"], QUANT_ORDER, ordered=True)
quality_cols = [c for c in df.columns if c.startswith("quality_")]


def short(col: str) -> str:
    return col.replace("quality_", "").replace("bertscore_", "bs_")


def draw_table(sub: pd.DataFrame, task: str) -> None:
    cols = [c for c in quality_cols if sub[c].notna().any()]
    sub = sub.sort_values(["family", "size_b", "quantization"])

    row_labels = [f"{r.family} {r.size_b}B {r.quantization}" for r in sub.itertuples()]
    values = sub[cols].to_numpy(dtype=float)

    text = [[f"{v:.3f}" if not np.isnan(v) else "–" for v in row] + [str(int(n))]
            for row, n in zip(values, sub["n_samples"])]

    cmap = plt.get_cmap("RdYlGn")
    colours = []
    norm = np.zeros_like(values)
    for j, c in enumerate(cols):
        col = values[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        n = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
        norm[:, j] = 1 - n if c in LOWER_IS_BETTER else n
    for i in range(len(sub)):
        colours.append([cmap(norm[i, j]) if not np.isnan(values[i, j]) else "white"
                        for j in range(len(cols))] + ["white"])

    col_labels = [short(c) for c in cols] + ["n"]

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
    ax.set_title(f"Correctness metrics – {task}", fontweight="bold", pad=12)

    fig.savefig(OUT_DIR / f"correctness_{task}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


for task, sub in df.groupby("task_type"):
    draw_table(sub, task)
    print(f"saved {OUT_DIR / f'correctness_{task}.png'}")