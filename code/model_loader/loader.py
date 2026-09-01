from typing import NamedTuple
from vllm import LLM, SamplingParams
from project_types.project_types import ModelConfig, Quantization
from functools import lru_cache


class LoadedModel(NamedTuple):
    llm: LLM
    label: str


_INT4_SUFFIX: dict[str, str] = {
    "llama3":  "-GPTQ-Int4",
    "mistral": "-GPTQ-Int4",
    "phi3":    "-GPTQ-Int4",
}

def _resolve_int4_model_id(model_id: str, family: str) -> str:
    suffix = _INT4_SUFFIX.get(family, "-GPTQ-Int4")
    if suffix.lower() in model_id.lower():
        return model_id
    return model_id.rstrip("/") + suffix
 
 
class LoadedModel(NamedTuple):
    llm: "LLM"
    label: str


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
 
    elif quant == Quantization.INT8:
        llm = LLM(
            model=model_id,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="float16",
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.70,
        )
 
    elif quant == Quantization.INT4:
        int4_id = _resolve_int4_model_id(model_id, config.family)
        llm = LLM(
            model=int4_id,
            quantization="gptq",
            dtype="float16",
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.90,
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