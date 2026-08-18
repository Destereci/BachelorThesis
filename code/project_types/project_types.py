from dataclasses import dataclass, field
from enum import Enum


class Quantization(str, Enum):

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class TaskType(str, Enum):

    SUMMARIZATION = "summarization"
    CODE = "code"
    JSON = "json"


@dataclass
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
    completion: str
    energy: PhaseEnergy
    latency_s: float
    quality_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    #TODO: define EQ score here
    

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    samples: list[SampleResult] = field(default_factory=list)

    mean_quality: dict[str, float] = field(default_factory=dict)
    mean_joules_per_output_token: float = 0.0
    mean_joules_per_input_token: float = 0.0
    mean_eq_score: float = 0.0
    mean_output_tokens: float = 0.0
    mean_input_tokens: float = 0.0
    total_joules: float = 0.0
    flops_per_task: float = 0.0

    def compute_aggregates(self) -> None:
        pass  # TODO: implement aggregate computation