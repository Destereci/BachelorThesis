from typing import NamedTuple
from vllm import LLM, SamplingParams
from types.project_types import ModelConfig, Quantization


class LoadedModel(NamedTuple):
    llm: LLM
    label: str


# Maps (family, quantization) → pre-quantized HF checkpoint
# These are loaded directly — no runtime quantization step
PREQUANTIZED_CHECKPOINTS: dict[tuple[str, str], str] = {
    ("mistral", "int4"): "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    ("mistral", "int8"): "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    ("llama3",  "int4"): "bartowski/Meta-Llama-3-8B-Instruct-GPTQ",
    ("llama3",  "int8"): "bartowski/Meta-Llama-3-8B-Instruct-GPTQ",
    ("phi3",    "int4"): "bartowski/Phi-3-mini-4k-instruct-GPTQ",
    ("phi3",    "int8"): "bartowski/Phi-3-mini-4k-instruct-GPTQ",
}


def load_model(config: ModelConfig) -> LoadedModel:
    quant = config.quantization
    key = (config.family, quant.value)

    if quant == Quantization.FP16:
        llm = LLM(
            model=config.model_id,
            dtype="float16",
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.9,
        )

    elif quant in (Quantization.INT8, Quantization.INT4):
        if key not in PREQUANTIZED_CHECKPOINTS:
            raise ValueError(
                f"No pre-quantized checkpoint registered for {key}. "
                f"Add it to PREQUANTIZED_CHECKPOINTS."
            )
        checkpoint = PREQUANTIZED_CHECKPOINTS[key]
        llm = LLM(
            model=checkpoint,
            quantization="gptq",
            dtype="float16",
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.9,
        )

    else:
        raise ValueError(f"Unsupported quantization: {quant}")

    return LoadedModel(llm=llm, label=config.label)


def make_sampling_params(
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )