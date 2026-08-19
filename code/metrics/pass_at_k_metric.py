from __future__ import annotations
import subprocess
import os
import re
import tempfile

from metrics.base_metric import BaseMetric, register_metric

@register_metric("pass_at_k")
class PassAtKMetric(BaseMetric):

    

    def score_batch(self, generated, references, samples):
        results = []
        for gen, ref, sample in zip(generated, references, samples):
            passed = self._execute(gen, sample)
            results.append(
                {
                    "primary": 1.0 if passed else 0.0,
                    "pass_at_1": 1.0 if passed else 0.0,
                }
            )
        return results


    def _execute(self, code: str, sample: dict) -> bool:

        code = re.sub(r"```(?:python)?", "", code).strip()

        test_code = sample.get("test_code", "")
        entry_point = sample.get("entry_point", "")
        full_code = f"{code}\n\n{test_code}\n"

        if entry_point:
            full_code += f"\ncheck({entry_point})\n"

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
            temp_file.write(full_code)
            temp_file_name=temp_file.name

        try:
            result = subprocess.run(
                ["python", temp_file_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.unlink(temp_file_name)
