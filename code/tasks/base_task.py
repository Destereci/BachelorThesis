from abc import ABC, abstractmethod

from types.project_types import TaskType


_TASK_REGISTRY: dict[TaskType, type["BaseTask"]] = {}

def register_task(task_type: TaskType):
    def decorator(cls: type[BaseTask]) -> type[BaseTask]:
        _TASK_REGISTRY[task_type] = cls
        return cls
    return decorator

def get_task(task_type: TaskType, dataset_name: str, split: str, max_samples: int) -> "BaseTask":
    return _TASK_REGISTRY[task_type](dataset_name, split, max_samples)


class BaseTask(ABC):

    def __init__(self, dataset_name: str, split: str, max_samples: int):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self._data: list[dict] = []

    @abstractmethod
    def load_dataset(self) -> None:
        pass

    @abstractmethod
    def format_prompt(self, sample: dict) -> str:
        pass

    @abstractmethod
    def get_reference(self, sample: dict) -> str:
        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        pass

    def __iter__(self):
        if not self._data:
            self.load_dataset()
        return iter(self._data)

    def __len__(self):
        if not self._data:
            self.load_dataset()
        return len(self._data)