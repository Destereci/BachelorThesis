import json
from pathlib import Path

RESULTS_DIR = Path("results")          # your results folder
BACKUP_DIR = RESULTS_DIR / "_backup_old_scoring"
BACKUP_DIR.mkdir(exist_ok=True)

for path in sorted(RESULTS_DIR.glob("*.json")):
    data = json.loads(path.read_text())
    samples = data["samples"]
    n = len(samples)
    if n == 0:
        continue

    (BACKUP_DIR / path.name).write_text(json.dumps(data))  # keep the original

    # 1) invalidate scores of empty generations
    for s in samples:
        q = s["quality_scores"]
        is_empty = bool(q.get("empty")) or s["energy"]["decode_tokens"] == 0
        if is_empty:
            for k in list(q):
                q[k] = None
            q["empty"] = True

    # 2) recompute quality aggregates (same logic as compute_aggregates)
    n_empty = sum(1 for s in samples if s["quality_scores"].get("empty"))
    mq = {"empty": n_empty / n, "n_valid": n - n_empty}
    keys = {k for s in samples for k in s["quality_scores"]} - {"empty"}
    for k in keys:
        vals = [float(s["quality_scores"][k]) for s in samples
                if s["quality_scores"].get(k) is not None]
        mq[k] = sum(vals) / len(vals) if vals else float("nan")
        mq[f"{k}_penalized"] = sum(vals) / n
    if "primary_penalized" in mq:
        mq["primary"] = mq["primary_penalized"]
    data["aggregates"]["mean_quality"] = mq

    # 3) energy per output token without zero-token samples
    jpo = [s["energy"]["joules_per_output_token"] for s in samples
           if s["energy"]["decode_tokens"] > 0]
    data["aggregates"]["mean_joules_per_output_token"] = sum(jpo) / len(jpo) if jpo else 0.0

    path.write_text(json.dumps(data, indent=2))
    print(f"fixed {path.name}: {n_empty}/{n} empty")