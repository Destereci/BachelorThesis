from datasets import load_dataset
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType

@register_task(TaskType.SUMMARIZATION)
class SummarizationTask(BaseTask):

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