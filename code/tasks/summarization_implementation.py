from datasets import load_dataset
from tasks.base_task import BaseTask, register_task
from project_types.project_types import TaskType

@register_task(TaskType.SUMMARIZATION)
class SummarizationTask(BaseTask):

    metric_name = "bertscore"
    stop_sequence = ["\n"]

    def load_dataset(self) -> None:
        if self.dataset_name == "xsum":
            ds = load_dataset("EdinburghNLP/xsum", split=self.split)
            ds = ds.shuffle(seed=self.seed)
            n = len(ds) if self.max_samples is None else min(self.max_samples, len(ds))
            self._data = [
                {
                    "id": str(row["id"]), 
                    "input": row["document"],
                    "reference": row["summary"],
                }
                for row in ds.select(range(n))
            ]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


    def format_prompt(self, sample: dict) -> str:
        return (
            "Summarize the following article in exactly one sentence. "
            "Reply with only the summary and nothing else.\n\n"
            f"Article:\n{sample['input']}"
        )


    def get_reference(self, sample: dict) -> str:
        return sample["reference"]