

import json
import os

from datasets import load_dataset


class JsonGenTask():
    def __init__(self, dataset_name: str, split: str, max_samples: int):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples

    metric_name = "json_validity"

    def load_dataset(self) -> None:
        if os.path.exists(f"{self.dataset_name}.json"):
            with open(f"{self.dataset_name}.json", "r") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            ds = load_dataset(self.dataset_name, split=self.split)
            rows = list(ds)

        self._data = rows[:self.max_samples]

    def format_prompt(self, sample: dict) -> str:
        return (
            f"{sample['instruction']}\n\n"
            f"Text:\n{sample['input_text']}\n\n"
            "Return only valid JSON, no explanation or markdown fences."
        )

    def get_reference(self, sample: dict) -> str:
        return json.dumps(sample["reference_json"])