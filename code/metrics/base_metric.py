from abc import ABC, abstractmethod



_METRIC_REGISTRY: dict[str, type["BaseMetric"]] = {}

def register_metric(metric_name: str):
    def decorator(cls: type["BaseMetric"]) -> type["BaseMetric"]:
        _METRIC_REGISTRY[metric_name] = cls
        return cls
    return decorator

def get_metric(metric_name: str) -> "BaseMetric":
    return _METRIC_REGISTRY[metric_name]

class BaseMetric(ABC):

    @abstractmethod
    def score_batch(
        self,
        generated: list[str],
        references: list[str],
        samples: list[dict],
    ) -> list[dict[str, float]]:
        ...

    def score_single(
        self, generated: str, references: str, sample: dict
    ) -> dict[str, float]:
        return self.score_batch([generated], [references], [sample])[0]