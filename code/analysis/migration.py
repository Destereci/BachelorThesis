
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")

# A unique backup directory for this run; never overwrite older backups.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = RESULTS_DIR / f"_backup_migration_{timestamp}"
BACKUP_DIR.mkdir(parents=True, exist_ok=False)


def get_total_joules(sample):
    """Return total measured energy for one request."""
    energy = sample["energy"]

    if energy.get("total_joules") is not None:
        return float(energy["total_joules"])

    return (
        float(energy["prefill_joules"])
        + float(energy["generation_joules"])
    )


def get_token_counts(sample):
    """Read token counts from the sample, falling back to energy fields."""
    energy = sample["energy"]

    input_tokens = sample.get("prefill_tokens")
    output_tokens = sample.get("decode_tokens")
    

    if input_tokens is None or output_tokens is None:
        raise ValueError(
            "Missing input/output token counts; refusing to migrate this file."
        )

    return int(input_tokens), int(output_tokens)


for path in sorted(RESULTS_DIR.glob("*.json")):
    # Read and validate everything before changing the result file.
    original_text = path.read_text(encoding="utf-8")
    data = json.loads(original_text)
    samples = data["samples"]

    if not samples:
        print(f"Skipped {path.name}: no samples")
        continue

    # Prepare token and energy data. If a required field is missing,
    # the file is left unchanged.
    try:
        rows = []
        for sample in samples:
            input_tokens, output_tokens = get_token_counts(sample)
            rows.append({
                "energy": get_total_joules(sample),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Skipped {path.name}: {exc}")
        continue

    # Preserve the exact file contents before changing anything.
    backup_path = BACKUP_DIR / path.name
    backup_path.write_text(original_text, encoding="utf-8")

    n = len(samples)
    total_energy = sum(row["energy"] for row in rows)
    total_input_tokens = sum(row["input_tokens"] for row in rows)
    total_output_tokens = sum(row["output_tokens"] for row in rows)
    total_processed_tokens = total_input_tokens + total_output_tokens

    # 1. Invalidate quality scores for empty generations.
    for sample, row in zip(samples, rows):
        quality = sample["quality_scores"]
        is_empty = (
            bool(quality.get("empty"))
            or row["output_tokens"] == 0
        )

        if is_empty:
            for key in list(quality):
                quality[key] = None
            quality["empty"] = True

    # 2. Recompute quality aggregates.
    n_empty = sum(
        bool(sample["quality_scores"].get("empty"))
        for sample in samples
    )

    mean_quality = {
        "empty": n_empty / n,
        "n_valid": n - n_empty,
    }

    quality_keys = {
        key
        for sample in samples
        for key in sample["quality_scores"]
    } - {"empty"}

    for key in quality_keys:
        values = [
            float(sample["quality_scores"][key])
            for sample in samples
            if sample["quality_scores"].get(key) is not None
        ]

        mean_quality[key] = (
            sum(values) / len(values) if values else None
        )
        mean_quality[f"{key}_penalized"] = sum(values) / n

    if "primary_penalized" in mean_quality:
        mean_quality["primary"] = mean_quality["primary_penalized"]

    aggregates = data.setdefault("aggregates", {})
    aggregates["mean_quality"] = mean_quality

    # 3. Energy per request: includes empty generations.
    aggregates["total_joules"] = total_energy
    aggregates["n_samples"] = n
    aggregates["mean_joules_per_request"] = total_energy / n

    # 4. Energy per processed token: ratio of total energy to total tokens.
    # Empty generations still contribute their energy and input tokens.
    aggregates["total_input_tokens"] = total_input_tokens
    aggregates["total_output_tokens"] = total_output_tokens
    aggregates["total_processed_tokens"] = total_processed_tokens

    aggregates["mean_joules_per_processed_token"] = (
        total_energy / total_processed_tokens
        if total_processed_tokens > 0
        else None
    )

    # 5. Retain the existing mean energy per output token metric.
    # This excludes zero-output samples because their ratio is undefined.
    per_output_token = [
        row["energy"] / row["output_tokens"]
        for row in rows
        if row["output_tokens"] > 0
    ]

    aggregates["mean_joules_per_output_token"] = (
        sum(per_output_token) / len(per_output_token)
        if per_output_token
        else None
    )

    # Write only after all calculations have succeeded.
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(
        f"Migrated {path.name}: "
        f"{n_empty}/{n} empty, "
        f"{aggregates['mean_joules_per_request']:.4f} J/request, "
        f"{aggregates['mean_joules_per_processed_token']} J/token"
    )

print(f"\nBackups saved to: {BACKUP_DIR}")
