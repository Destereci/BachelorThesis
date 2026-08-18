import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_types.project_types import ModelConfig, ExperimentConfig, Quantization, TaskType
from runner import run_experiment



MODEL_CATALOGUE: dict[str, dict] = {
    "llama3_8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "llama3",
        "size_b": 8.0,
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "size_b": 7.0,
    },
    "phi3_mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "family": "phi3",
        "size_b": 3.8,
    },
}

QUANTIZATIONS = [Quantization.FP16, Quantization.INT8, Quantization.INT4]

DEFAULT_DATASETS: dict[str, str] = {
    "summarization": "xsum",
    "code":          "humaneval",
    "json_gen":      "data/json_prompts.jsonl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one thesis experiment sweep.")
    p.add_argument("--task",        required=True,  choices=list(DEFAULT_DATASETS.keys()))
    p.add_argument("--dataset",     default=None,   help="Override default dataset")
    p.add_argument("--models",      default="all",  help="'all' or comma-sep model keys")
    p.add_argument("--quants",      default="all",  help="'all' or fp16,int8,int4")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-tokens",  type=int, default=512)
    p.add_argument("--output-dir",  default="results")
    p.add_argument("--split",       default="test")
    return p.parse_args()


def main():
    args = parse_args()

    task_type = TaskType(args.task)
    dataset   = args.dataset or DEFAULT_DATASETS[args.task]

    if args.models == "all":
        selected_models = list(MODEL_CATALOGUE.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    if args.quants == "all":
        selected_quants = QUANTIZATIONS
    else:
        quant_map = {"fp16": Quantization.FP16, "int8": Quantization.INT8, "int4": Quantization.INT4}
        selected_quants = [quant_map[q.strip()] for q in args.quants.split(",")]

    print(f"\n{'='*60}")
    print(f"  Task:     {task_type.value}")
    print(f"  Dataset:  {dataset}")
    print(f"  Models:   {selected_models}")
    print(f"  Quants:   {[q.value for q in selected_quants]}")
    print(f"  Samples:  {args.max_samples}")
    print(f"{'='*60}\n")

    results_summary = []

    for model_key in selected_models:
        if model_key not in MODEL_CATALOGUE:
            print(f"[SKIP] Unknown model key: {model_key}")
            continue

        spec = MODEL_CATALOGUE[model_key]

        for quant in selected_quants:
            model_cfg = ModelConfig(
                model_id=spec["model_id"],
                family=spec["family"],
                size_b=spec["size_b"],
                quantization=quant,
            )
            exp_cfg = ExperimentConfig(
                model_config=model_cfg,
                task_type=task_type,
                dataset_name=dataset,
                dataset_split=args.split,
                max_samples=args.max_samples,
                max_new_tokens=args.max_tokens,
                output_dir=args.output_dir,
            )

            print(f"\n>> Running: {model_cfg.label} | {task_type.value} | {dataset}")
            try:
                result = run_experiment(exp_cfg)
                summary = {
                    "model":       model_cfg.label,
                    "size_b":      model_cfg.size_b,
                    "quant":       quant.value,
                    "task":        task_type.value,
                    "quality":     result.mean_quality,
                    "j_per_tok":   round(result.mean_joules_per_output_token, 4),
                    "eq_score":    round(result.mean_eq_score, 4),
                    "out_tokens":  round(result.mean_output_tokens, 1),
                }
                results_summary.append(summary)
                _print_summary_row(summary)
            except Exception as e:
                print(f"   [ERROR] {e}")

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'Quant':<6} {'Primary Q':>10} {'J/tok':>8} {'EQ':>8}")
    print(f"  {'-'*60}")
    for s in results_summary:
        primary = s["quality"].get("primary", 0)
        print(f"  {s['model']:<28} {s['quant']:<6} {primary:>10.3f} {s['j_per_tok']:>8.4f} {s['eq_score']:>8.4f}")


def _print_summary_row(s: dict) -> None:
    primary = s["quality"].get("primary", 0)
    print(
        f"   OK  | quality={primary:.3f} | "
        f"J/tok={s['j_per_tok']:.4f} | EQ={s['eq_score']:.4f} | "
        f"out_tokens={s['out_tokens']:.0f}"
    )


if __name__ == "__main__":
    main()