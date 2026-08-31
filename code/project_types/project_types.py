from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Quantization(str, Enum):

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class TaskType(str, Enum):

    SUMMARIZATION = "summarization"
    CODE = "code"
    JSON = "json"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    family: str
    size_b: float
    quantization: Quantization
    max_model_len: int = 2048

    @property
    def label(self) -> str:
        return f"{self.family}_{self.size_b}B_{self.quantization.value}"


@dataclass
class ExperimentConfig:
    model_config: ModelConfig
    task_type: TaskType
    dataset_name: str
    dataset_split: str = "test"
    max_samples : int = 100
    max_new_tokens: int = 512
    temperature: float = 0.0
    seed: int = 42
    output_dir: str = "results"


@dataclass
class PhaseEnergy:
    prefill_joules: float
    generation_joules: float
    prefill_tokens: int
    decode_tokens: int


    @property
    def total_joules(self) -> float:
        return self.prefill_joules + self.generation_joules


    @property
    def joules_per_output_token(self) -> float:
        if self.decode_tokens == 0:
            return 0.0
        return self.total_joules / self.decode_tokens

    @property
    def joules_per_input_token(self) -> float:
        if self.prefill_tokens == 0:
            return 0.0
        return self.prefill_joules / self.prefill_tokens


@dataclass
class SampleResult:
    sample_id: str
    prompt: str
    reference: str
    generated: str
    energy: PhaseEnergy
    latency_s: float
    quality_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    samples: list[SampleResult] = field(default_factory=list)

    mean_quality: dict[str, float] = field(default_factory=dict)
    mean_joules_per_output_token: float = 0.0
    mean_joules_per_input_token: float = 0.0
    mean_output_tokens: float = 0.0
    mean_input_tokens: float = 0.0
    total_joules: float = 0.0
    #flops_per_task: float = 0.0
    #eq_score: float = 0.0

    def compute_aggregates(self) -> None:

        if not self.samples:
            return
        n = len(self.samples)
 
        self.mean_joules_per_output_token = sum(
            s.energy.joules_per_output_token for s in self.samples
        ) / n
        self.mean_joules_per_input_token = sum(
            s.energy.joules_per_input_token for s in self.samples
        ) / n
        self.total_joules = sum(s.energy.total_joules for s in self.samples)
        self.mean_output_tokens = sum(s.energy.decode_tokens for s in self.samples) / n
        self.mean_input_tokens = sum(s.energy.prefill_tokens for s in self.samples) / n
 
 
        all_keys = set()
        for s in self.samples:
            all_keys.update(s.quality_scores.keys())
        for key in all_keys:
            vals = [s.quality_scores[key] for s in self.samples if key in s.quality_scores]
            self.mean_quality[key] = sum(vals) / len(vals) if vals else 0.0