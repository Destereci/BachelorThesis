import jsonschema
import re
import json
from metrics.base_metric import BaseMetric, register_metric

@register_metric("json_validity")
class JsonValidityMetric(BaseMetric):

    def score_batch(self, generated, references, samples):

        results = []
        for gen, ref, sample in zip(generated, references, samples):
            gen_clean = re.sub(r"```(?:json)?\n?", "", gen).strip()

            parseable = 0.0
            schema_valid = 0.0
            field_f1 = 0.0

            try:
                gen_obj = json.loads(gen_clean)
                parseable = 1.0
                if "schema" in sample:
                    try:
                        jsonschema.validate(gen_obj, sample["schema"])
                        schema_valid = 1.0
                    except jsonschema.ValidationError:
                        schema_valid = 0.0
                else:
                    schema_valid = parseable

                ref_obj = json.loads(ref) if isinstance(ref, str) else ref
                field_f1 = self.compute_field_f1(gen_obj, ref_obj)
            except (json.JSONDecodeError, ValueError):
                pass

            results.append({
                "primary": parseable,
                "parseable": parseable,
                "schema_valid": schema_valid,
                "field_f1": field_f1,
            })
        return results

    @staticmethod
    def compute_field_f1(gen: dict, ref: dict) -> float:
        if not isinstance(gen, dict) or not isinstance(ref, dict):
                return 0.0
        ref_keys = set(ref.keys())
        gen_keys = set(gen.keys())
        if not ref_keys:
            return 1.0
        tp = sum(
            1 for k in ref_keys
            if k in gen_keys and str(gen[k]).strip() == str(ref[k]).strip()
        )
        precision = tp / len(gen_keys) if gen_keys else 0.0
        recall    = tp / len(ref_keys)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)