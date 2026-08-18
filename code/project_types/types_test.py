from project_types import ModelConfig, ExperimentConfig, TaskType, Quantization, PhaseEnergy, SampleResult


model_config = ModelConfig(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",    
    family="llama3",
    size_b=8.0,
    quantization=Quantization.FP16,
)

experiment_config = ExperimentConfig(
    model=model_config,
    task_type=TaskType.SUMMARIZATION,
    dataset_name="cnn_dailymail",
)

phase_energy = PhaseEnergy(
    prefill_joules=100.0,
    decode_joules=200.0,
    prefill_tokens=10,
    decode_tokens=20,
)

sample_result = SampleResult(
    sample_id="sample_001",
    prompt="Summarize the following article...",
    reference="This is the reference summary.",
    generated="This is the generated summary.",
    energy=phase_energy,
    latency_s=1.0,
    quality_scores={"rouge": 0.85, "bleu": 0.75},
    metadata={"key1": "value1", "key2": "value2"},
)
    



print(f"Model Config: {model_config}")
print(f"Experiment Config: {experiment_config}")
print(f"Phase Energy: {phase_energy}")
print(f"Sample Result: {sample_result}")