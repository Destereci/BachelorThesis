import time
import uuid
from pathlib import Path
import json

import metrics.base_metric 
from metrics.base_metric import get_metric
from project_types.project_types import ExperimentConfig, ExperimentResult, PhaseEnergy, SampleResult
from tasks.base_task import get_task
import tasks.json_implementation, tasks.summarization_implementation, tasks.code_implementation
import metrics.bert_score_metric, metrics.json_validity_metric, metrics.pass_at_k_metric
from energy import Energy_Monitor
from model_loader.loader import load_model, make_sampling_params





def run_experiment(config: ExperimentConfig) -> ExperimentResult:

    # Setup

    task = get_task(
        config.task_type,
        config.dataset_name,
        config.dataset_split,
        config.max_samples
    )
    task.load_dataset()
    metric = get_metric(task.metric_name)
    monitor = Energy_Monitor()
    loaded = load_model(config.model_config)
    llm = loaded.llm
    sampling = make_sampling_params(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        seed=config.seed
    )

    result = ExperimentResult(config=config)

    # Loop

    all_generated: list[str] = []
    all_references: list[str] = []
    all_samples: list[dict] = []

    for sample in task:
        prompt = task.format_prompt(sample)
        reference = task.get_reference(sample)

        monitor.start()
        t0 = time.perf_counter()

        outputs = llm.generate([prompt], sampling)
        monitor.mark_prefill_end()

        latency_s = time.perf_counter() - t0
        energy: PhaseEnergy = monitor.stop()

        output = outputs[0]
        generated_text = output.outputs[0].text
        input_tokens_count = len(output.prompt_token_ids)
        output_tokens_count = len(output.outputs[0].token_ids)

        energy = PhaseEnergy(
            prefill_joules=energy.prefill_joules,
            generation_joules=energy.generation_joules,
            prefill_tokens=input_tokens_count,
            decode_tokens=output_tokens_count
        )


        all_generated.append(generated_text)
        all_references.append(reference)
        all_samples.append(sample)

        result.samples.append(SampleResult(
            sample_id=str(sample.get("id", uuid.uuid4())),
            prompt=prompt,
            reference=reference,
            generated=generated_text,
            energy=energy,
            latency_s=latency_s
        ))

    quality_scores = metric.score_batch(all_generated, all_references, all_samples)
    for i, scores in enumerate(quality_scores):
        result.samples[i].quality_scores = scores

    result.compute_aggregates()

    _save_result(result, config.output_dir)

    return result


def _save_result(result: ExperimentResult, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{result.config.model_config.label}__{result.config.task_type.value}__{result.config.dataset_name}.json"
    path = Path(output_dir) / filename

    payload = {
        "config": {
            "model":        result.config.model_config.label,
            "model_id":     result.config.model_config.model_id,
            "family":       result.config.model_config.family,
            "size_b":       result.config.model_config.size_b,
            "quantization": result.config.model_config.quantization.value,
            "task_type":    result.config.task_type.value,
            "dataset":      result.config.dataset_name,
            "max_samples":  result.config.max_samples,
        },
        "aggregates": {
            "mean_quality":                  result.mean_quality,
            "mean_joules_per_output_token":  result.mean_joules_per_output_token,
            "mean_joules_per_input_token":   result.mean_joules_per_input_token,
            "mean_output_tokens":            result.mean_output_tokens,
            "mean_input_tokens":             result.mean_input_tokens,
            "total_joules":                  result.total_joules,
            "n_samples":                     len(result.samples),
        },
        "samples": [
            {
                "id":             s.sample_id,
                "generated":      s.generated,
                "reference":      s.reference,
                "quality_scores": s.quality_scores,
                "energy": {
                    "prefill_joules":          s.energy.prefill_joules,
                    "decode_joules":           s.energy.generation_joules,
                    "total_joules":            s.energy.total_joules,
                    "joules_per_output_token": s.energy.joules_per_output_token,
                    "joules_per_input_token":  s.energy.joules_per_input_token,
                    "prefill_tokens":          s.energy.prefill_tokens,
                    "decode_tokens":           s.energy.decode_tokens,
                },
                "latency_s": s.latency_s,
            }
            for s in result.samples
        ],
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[runner] Saved → {path}")




