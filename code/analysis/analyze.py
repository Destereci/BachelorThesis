from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def load_result_file(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_single(data: dict) -> dict:
    
    cfg = data["config"]
    agg = data["aggregates"]
    qual = agg["mean_quality"]

    return {
        # config / levers
        "model":            cfg["model"],
        "family":           cfg["family"],
        "size_b":           cfg["size_b"],
        "quantization":     cfg["quantization"],
        "task_type":        cfg["task_type"],
        "dataset":          cfg["dataset"],
        "n_samples":        agg["n_samples"],
        # token counts
        "mean_input_tokens":  agg["mean_input_tokens"],
        "mean_output_tokens": agg["mean_output_tokens"],
        # energy
        "j_per_output_token": agg["mean_joules_per_output_token"],
        "j_per_input_token":  agg["mean_joules_per_input_token"],
        "total_joules":       agg["total_joules"],
        # quality
        "quality_primary":    qual.get("primary", 0.0),
        **{f"quality_{k}": v for k, v in qual.items() if k != "primary"},
    }


def analyze_samples(data: dict) -> pd.DataFrame:

    rows = []
    for s in data["samples"]:
        row = {
            "sample_id":          s["id"],
            "quality_primary":    s["quality_scores"].get("primary", 0.0),
            "prefill_joules":     s["energy"]["prefill_joules"],
            "decode_joules":      s["energy"]["decode_joules"],
            "total_joules":       s["energy"]["total_joules"],
            "j_per_output_token": s["energy"]["joules_per_output_token"],
            "j_per_input_token":  s["energy"]["joules_per_input_token"],
            "prefill_tokens":     s["energy"]["prefill_tokens"],
            "decode_tokens":      s["energy"]["decode_tokens"],
            "latency_s":          s["latency_s"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_file(path: Path, output_dir: Path) -> dict:
    
    data = load_result_file(path)
    summary = summarize_single(data)
    sample_df = analyze_samples(data)

    # Print report
    print(f"\n{'='*60}")
    print(f"  {summary['model']}  |  {summary['task_type']}  |  {summary['dataset']}")
    print(f"{'='*60}")
    print(f"  Samples:          {summary['n_samples']}")
    print(f"  Quality (primary):{summary['quality_primary']:.4f}")
    print(f"  J / output token: {summary['j_per_output_token']:.4f}")
    print(f"  J / input token:  {summary['j_per_input_token']:.4f}")
    print(f"  Total joules:     {summary['total_joules']:.2f}")
    print(f"  Mean input tok:   {summary['mean_input_tokens']:.1f}")
    print(f"  Mean output tok:  {summary['mean_output_tokens']:.1f}")
    print(f"\n  Per-sample breakdown:")
    print(sample_df[["sample_id", "quality_primary", "j_per_output_token",
                      "decode_tokens", "prefill_tokens", "latency_s"
                      ]].to_string(index=False, float_format="{:.4f}".format))

    # Save per-sample CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem  # e.g. phi3_3.8B_fp16__code__humaneval
    csv_path = output_dir / f"{stem}__samples.csv"
    sample_df.to_csv(csv_path, index=False)
    print(f"\n  Saved per-sample CSV → {csv_path}")

    return summary


def analyze_directory(results_dir: Path, task_filter: str | None, output_dir: Path) -> pd.DataFrame:
    
    summaries = []
    for path in sorted(results_dir.glob("*.json")):
        data = load_result_file(path)
        cfg = data["config"]
        if task_filter and task_filter != "all" and cfg["task_type"] != task_filter:
            continue
        summary = analyze_file(path, output_dir)
        summaries.append(summary)

    if not summaries:
        print("No matching result files found.")
        return pd.DataFrame()

    df = pd.DataFrame(summaries)
    df["quantization"] = pd.Categorical(
        df["quantization"], categories=["fp16", "int8", "int4"], ordered=True
    )
    df = df.sort_values(["family", "size_b", "quantization", "task_type"])

    # Save full summary CSV
    summary_path = output_dir / "summary_all.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved full summary → {summary_path}")

    return df


def main():
    p = argparse.ArgumentParser(description="Analyze experiment result files.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",        type=Path, help="Analyze a single result JSON")
    group.add_argument("--results-dir", type=Path, help="Analyze all JSONs in a directory")
    p.add_argument("--task",       default="all", help="Filter by task type (with --results-dir)")
    p.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = p.parse_args()

    if args.file:
        analyze_file(args.file, args.output_dir)
    else:
        df = analyze_directory(args.results_dir, args.task, args.output_dir)
        if not df.empty:
            print(f"\n{'='*60}")
            print("  SUMMARY TABLE")
            print(f"{'='*60}")
            print(df[["model", "quantization", "task_type",
                       "quality_primary", "j_per_output_token", "total_joules"
                       ]].to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()