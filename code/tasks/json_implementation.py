import json
import os
from random import sample

from datasets import load_dataset
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType


@register_task(TaskType.JSON)
class JsonGenTask(BaseTask):

    metric_name = "json_validity"

    def load_dataset(self) -> None:
        ds = load_dataset("epfl-dlab/JSONSchemaBench")
        self._data = [
            {
                "id": row["unique_id"],
                "schema": row["json_schema"],  
            }
            for row in ds.select(range(self._n(len(ds))))
        ]

    def format_prompt(self, sample: dict) -> str:
        return (
            "Generate a JSON object that satisfies the following schema.\n"
            f"Schema: {sample['schema']}\n"
            "Output only the JSON object."
        )

    def get_reference(self, sample: dict) -> str:
        return "" 