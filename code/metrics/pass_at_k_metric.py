from __future__ import annotations
import subprocess
import os
import re
import tempfile
import sys

from matplotlib.pylab import sample

from metrics.base_metric import BaseMetric, register_metric
    
STOP_SEQUENCES = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```"]

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

        code = self._truncate(code)

        test_code = sample.get("test_code", "")
        entry_point = sample.get("entry_point", "")
        full_code = sample["input"] + "\n" + code
        full_code += f"\n\n{test_code}\n"

        if entry_point:
            full_code += f"\ncheck({entry_point})\n"

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(full_code)
            temp_file_name=temp_file.name

        #print("="*40)
        #print(full_code)
        #print("="*40)

        try:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", temp_file_name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )
            #print("STDOUT:", result.stdout)
            #print("STDERR:", result.stderr)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.unlink(temp_file_name)

    @staticmethod
    def _truncate(code: str) -> str:
        code = re.sub(r"^\s*```(?:python)?\s*\n", "", code)
        ends = [code.find(s) for s in STOP_SEQUENCES if s in code]
        return (code[:min(ends)] if ends else code).rstrip()