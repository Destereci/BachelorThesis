from __future__ import annotations
from datasets import load_dataset, get_dataset_config_names
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType


@register_task(TaskType.JSON)
class JsonGenTask(BaseTask):

    metric_name = "json_validity"

    def load_dataset(self) -> None:
        if self.dataset_name == "JSONSchemaBench":
            print(get_dataset_config_names("epfl-dlab/JSONSchemaBench"))
            ds = load_dataset("epfl-dlab/JSONSchemaBench", "Github_easy", split=self.split)
            print(ds.column_names)
            print(ds[0])
            ds = ds.filter(lambda x: x["category"] in {"github_easy", "github_hard"})
            self._data = [
                {
                    "id": row["unique_id"],
                    "schema": row["json_schema"],  
                }
                for row in ds.select(range(self._n(len(ds))))
            ]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def format_prompt(self, sample: dict) -> str:
        return (
            "Generate a JSON object that satisfies the following schema.\n"
            f"Schema: {sample['schema']}\n"
            "Output only the JSON object."
        )

    def get_reference(self, sample: dict) -> str:
        return "" 