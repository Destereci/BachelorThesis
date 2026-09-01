from __future__ import annotations
import argparse
import json
from pathlib import Path
from itertools import product

import pandas as pd


# Core EQ formula 

def compute_eq_score(
    quality_quant: float,
    quality_fp16: float,
    energy_quant: float,
    energy_fp16: float,
) -> float | None:
    """ EQ = Δenergy - Δquality"""
    if quality_fp16 == 0 or energy_fp16 == 0:
        return None

    delta_quality = (quality_fp16 - quality_quant) / quality_fp16
    delta_energy  = (energy_fp16  - energy_quant)  / energy_fp16

    eq = delta_energy - delta_quality
    return round(max(-1.0, min(1.0, eq)), 6)


# File loading

def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_key_metrics(data: dict) -> dict:
    
    agg  = data["aggregates"]
    qual = agg["mean_quality"]
    cfg  = data["config"]
    return {
        "model":            cfg["model"],
        "family":           cfg["family"],
        "size_b":           cfg["size_b"],
        "quantization":     cfg["quantization"],
        "task_type":        cfg["task_type"],
        "dataset":          cfg["dataset"],
        "n_samples":        agg["n_samples"],
        "quality_primary":  qual.get("primary", 0.0),
        "j_per_output_token": agg["mean_joules_per_output_token"],
        "j_per_input_token":  agg["mean_joules_per_input_token"],
        "total_joules":       agg["total_joules"],
        "mean_output_tokens": agg["mean_output_tokens"],
        "mean_input_tokens":  agg["mean_input_tokens"],
    }


# Matching logic

def group_results(results_dir: Path, task_filter: str | None) -> dict:
    
    groups: dict[tuple, dict] = {}

    for path in sorted(results_dir.glob("*.json")):
        data = load_result(path)
        cfg = data["config"]

        if task_filter and task_filter != "all" and cfg["task_type"] != task_filter:
            continue

        key = (cfg["family"], cfg["size_b"], cfg["task_type"], cfg["dataset"])
        if key not in groups:
            groups[key] = {"fp16": None, "int8": None, "int4": None}

        quant = cfg["quantization"]
        if quant in groups[key]:
            groups[key][quant] = extract_key_metrics(data)

    return groups


# EQ computation

def compute_all_eq_scores(groups: dict) -> list[dict]:
    
    results = []

    for (family, size_b, task_type, dataset), variants in groups.items():
        baseline = variants.get("fp16")
        if baseline is None:
            print(f"  [SKIP] No FP16 baseline for ({family}, {size_b}B, {task_type}, {dataset})")
            continue

        for quant in ("int8", "int4"):
            quant_metrics = variants.get(quant)
            if quant_metrics is None:
                continue

            eq = compute_eq_score(
                quality_quant=quant_metrics["quality_primary"],
                quality_fp16=baseline["quality_primary"],
                energy_quant=quant_metrics["j_per_output_token"],
                energy_fp16=baseline["j_per_output_token"],
            )

            delta_quality = None
            delta_energy  = None
            if baseline["quality_primary"] != 0:
                delta_quality = round(
                    (baseline["quality_primary"] - quant_metrics["quality_primary"])
                    / baseline["quality_primary"], 6
                )
            if baseline["j_per_output_token"] != 0:
                delta_energy = round(
                    (baseline["j_per_output_token"] - quant_metrics["j_per_output_token"])
                    / baseline["j_per_output_token"], 6
                )

            results.append({
                # identification
                "family":       family,
                "size_b":       size_b,
                "task_type":    task_type,
                "dataset":      dataset,
                "quantization": quant,
                # baseline values
                "fp16_quality":  baseline["quality_primary"],
                "fp16_j_per_tok": baseline["j_per_output_token"],
                # quantized values
                "quant_quality":  quant_metrics["quality_primary"],
                "quant_j_per_tok": quant_metrics["j_per_output_token"],
                # deltas
                "delta_quality_pct": round(delta_quality * 100, 2) if delta_quality is not None else None,
                "delta_energy_pct":  round(delta_energy  * 100, 2) if delta_energy  is not None else None,
                # EQ score
                "eq_score": eq,
            })

    return results


# Output 

def save_eq_results(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # One JSON per family, task, dataset, with all quant comparisons grouped
    groups: dict[tuple, list] = {}
    for r in results:
        key = (r["family"], r["size_b"], r["task_type"], r["dataset"])
        groups.setdefault(key, []).append(r)

    for (family, size_b, task_type, dataset), comparisons in groups.items():
        filename = f"eq__{family}_{size_b}B__{task_type}__{dataset}.json"
        path = output_dir / filename
        with open(path, "w") as f:
            json.dump({"comparisons": comparisons}, f, indent=2)
        print(f"  Saved → {path}")

    # One CSV with everything
    df = pd.DataFrame(results)
    csv_path = output_dir / "eq_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved summary → {csv_path}")


def print_eq_report(results: list[dict]) -> None:
    if not results:
        print("No EQ scores computed, check that FP16 and quantized result files exist.")
        return

    print(f"\n{'='*70}")
    print("  EQ SCORES  (EQ = Δenergy - Δquality, range [-1, 1])")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Quant':<6} {'Task':<15} {'ΔQuality%':>10} {'ΔEnergy%':>10} {'EQ':>8}")
    print(f"  {'-'*70}")

    for r in sorted(results, key=lambda x: (x["family"], x["task_type"], x["quantization"])):
        dq = f"{r['delta_quality_pct']:+.1f}%" if r["delta_quality_pct"] is not None else "  N/A"
        de = f"{r['delta_energy_pct']:+.1f}%"  if r["delta_energy_pct"]  is not None else "  N/A"
        eq = f"{r['eq_score']:+.4f}"            if r["eq_score"]          is not None else "  N/A"
        label = f"{r['family']}_{r['size_b']}B"
        print(f"  {label:<20} {r['quantization']:<6} {r['task_type']:<15} {dq:>10} {de:>10} {eq:>8}")

    print(f"{'='*70}")
    print("  Interpretation:")
    print("    EQ > 0  → energy savings outweigh quality loss (efficient)")
    print("    EQ = 0  → savings and losses cancel out (neutral)")
    print("    EQ < 0  → quality loss outweighs energy savings (inefficient)")


def main():
    p = argparse.ArgumentParser(description="Compute EQ scores from experiment results.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--task",        default="all")
    p.add_argument("--output-dir",  type=Path, default=Path("results/eq_scores"))
    args = p.parse_args()

    print(f"Loading results from: {args.results_dir}")
    groups = group_results(args.results_dir, args.task)
    print(f"Found {len(groups)} unique (family, task, dataset) groups")

    results = compute_all_eq_scores(groups)
    print_eq_report(results)
    save_eq_results(results, args.output_dir)


if __name__ == "__main__":
    main()