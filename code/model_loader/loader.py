from typing import NamedTuple
from vllm import LLM
from types.project_types import ModelConfig, Quantization


class LoadedModel(NamedTuple):
    llm: "LLM"
    label: str


def load_model(config: ModelConfig) -> LoadedModel:

    quant = config.quantization
    model_id = config.model_id

    if quant == Quantization.FP16:
        llm = LLM(model=model_id, 
                  max_model_len=config.max_model_len, 
                  gpu_memory_utilization=0.9,
                  dtype="float16")

    elif quant == Quantization.INT8:
        llm = LLM(model=model_id, 
                  max_model_len=config.max_model_len, 
                  gpu_memory_utilization=0.9,
                  dtype="float16",
                  quantization="bitsandbytes",
                  load_format="bitsandbytes")

    elif quant == Quantization.INT4:
        llm = LLM(model=model_id, 
                  max_model_len=config.max_model_len, 
                  gpu_memory_utilization=0.9,
                  dtype="float16",
                  quantization="gptq")

    else:
        raise ValueError(f"Unsupported quantization: {quant}")

    return LoadedModel(llm=llm, label=config.label)