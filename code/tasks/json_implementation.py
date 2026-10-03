import json
import os

from datasets import load_dataset
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType


@register_task(TaskType.JSON)
class JsonGenTask(BaseTask):

    metric_name = "json_validity"

    def load_dataset(self) -> None:
        if os.path.exists(self.dataset_name):
            with open(self.dataset_name, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            ds = load_dataset(self.dataset_name, split=self.split)
            rows = list(ds)

        self._data = rows[:self.max_samples]

    def format_prompt(self, sample: dict) -> str:
        if "prompt" in sample:
            return "\n\n".join(m["content"] for m in sample["prompt"])

        if "json_schema" in sample:
            return f"Generate a JSON object that conforms to this schema:\n{sample['json_schema']}"

        raise KeyError(f"Unknown sample format: {list(sample.keys())}")

    def get_reference(self, sample: dict) -> str:
        return sample.get("completion", "")