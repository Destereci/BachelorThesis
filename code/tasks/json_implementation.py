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
        if os.path.exists(self.dataset_name):
            with open(self.dataset_name, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            ds = load_dataset(self.dataset_name, split=self.split)
            rows = list(ds)

        self._data = rows[:self.max_samples]

    def format_prompt(self, sample: dict) -> str:
        schema = sample.get("json_schema", sample.get("schema"))

        if schema is None:
            raise KeyError(f"No JSON schema found in sample: {list(sample.keys())}")

        schema_str = json.dumps(schema, separators=(",", ":"))

        if "prompt" in sample:
            prompt = sample["prompt"]

            if isinstance(prompt, list):
                task_parts  = [
                    m["content"]
                    for m in prompt
                    if m.get("role") == "user"
                ]
                task = "\n".join(task_parts)
            else:
                task = str(prompt)
        else:
            task = ""

        return (
            "Generate a JSON object that satisfies the following schema.\n"
            f"Schema: {schema_str}\n"
            f"Task: {task}\n"
            "Output only the JSON object."
        )

    def get_reference(self, sample: dict) -> str:
        reference = sample.get("completion", "")
        
        if isinstance(reference, (dict, list)):
            return json.dumps(reference)

        return reference