from __future__ import annotations
from datasets import load_dataset
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType


@register_task(TaskType.CODE)
class CodeTask(BaseTask):


    metric_name = "pass_at_k"

    def load_dataset(self) -> None:
        if self.dataset_name == "humaneval":
            ds = load_dataset("openai/openai_humaneval", split="test")
            self._data = [
                {
                    "id": row["task_id"],
                    "input": row["prompt"],
                    "reference": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
                for row in ds.select(range(min(self.max_samples, len(ds))))
            ]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


    def format_prompt(self, sample: dict) -> str:
        return ("Complete the following Python function. Return only the function body.\n\n" + sample["input"])


    def get_reference(self, sample: dict) -> str:
        return sample["reference"]
