from __future__ import annotations
from datasets import load_dataset, get_dataset_config_names
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType

SUBSETS = ["Github_easy", "Github_hard"]

@register_task(TaskType.JSON)
class JsonGenTask(BaseTask):

    metric_name = "json_validity"
    

    def load_dataset(self) -> None:
        if self.dataset_name == "JSONSchemaBench":
            self._data = []
            for subset in SUBSETS:
                ds = load_dataset("epfl-dlab/JSONSchemaBench", subset, split=self.split)
                ds = ds.shuffle(seed=self.seed)
                ds = ds.select(range(self._n(len(ds))))
                self._data.extend([
                    {
                        "id": f"{subset}/{row['unique_id']}",
                        "schema": row["json_schema"],
                        "subset": subset,
                    }
                    for row in ds
                ])
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