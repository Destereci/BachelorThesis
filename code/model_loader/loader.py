from typing import NamedTuple
from vllm import LLM, SamplingParams
from project_types.project_types import ModelConfig, Quantization
from functools import lru_cache


class LoadedModel(NamedTuple):
    llm: LLM
    label: str



PREQUANTIZED_CHECKPOINTS: dict[tuple[str, str], dict[str, str]] = {
    
    # INT4 GPTQ
    ("phi3",    "int4"): {"model_id": "ssuncheol/Phi-3-mini-128k-instruct-int4", "quant_format": "gptq"},
    ("mistral", "int4"): {"model_id": "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit", "quant_format": "gptq"},
    ("llama3",  "int4"): {"model_id": "study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int4", "quant_format": "gptq"},

    # INT8 W8A8 compressed-tensors
    ("phi3",    "int8"): {"model_id": "RedHatAI/Phi-3-mini-128k-instruct-quantized.w8a8", "quant_format": "compressed-tensors"},
    ("mistral", "int8"): {"model_id": "RedHatAI/Mistral-7B-Instruct-v0.3-quantized.w8a8", "quant_format": "compressed-tensors"},
    ("llama3",  "int8"): {"model_id": "RedHatAI/Meta-Llama-3-8B-Instruct-quantized.w8a8", "quant_format": "compressed-tensors"},

}
 


@lru_cache(maxsize=4)
def load_model(config: ModelConfig) -> LoadedModel:
    quant = config.quantization
    model_id = config.model_id
 
    if quant == Quantization.FP16:
        llm = LLM(
            model=model_id,
            dtype="float16",
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.90,
        )
 
    elif quant in (Quantization.INT8, Quantization.INT4):
        key = (config.family, quant.value)
        if key not in PREQUANTIZED_CHECKPOINTS:
            raise ValueError(
                f"No pre-quantized checkpoint for {key}. ")

        checkpoint = PREQUANTIZED_CHECKPOINTS[key]

        llm = LLM(
            model=checkpoint["model_id"],
            quantization=checkpoint["quant_format"],
            dtype="float16",
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.9,
        )
 
 
    else:
        raise ValueError(f"Unknown quantization: {quant}")
 
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