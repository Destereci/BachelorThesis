
from datasets import load_dataset

class SummarizationTask():
    def __init__(self, dataset_name: str, split: str, max_samples: int) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples

    metric_name = "bertscore"

    def load_dataset(self) -> None:
        if self.dataset_name == "xsum":
            ds = load_dataset("xsum", split=self.split)
            self._data = [
                {
                    "id": str(row["id"]), 
                    "input": row["document"],
                    "reference": row["summary"],
                }
                for row in ds.select(range(min(self.max_samples, len(ds))))
            ]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


    def format_prompt(self, sample: dict) -> str:
        return (
            "Summarize the following article in 2-3 sentences.\n\n"
            f"Article:\n{sample['input']}\n\nSummary:"
        )


    def get_reference(self, sample: dict) -> str:
        return sample["reference"]