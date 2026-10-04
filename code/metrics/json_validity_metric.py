import jsonschema
import re
import json
from metrics.base_metric import BaseMetric, register_metric

@register_metric("json_validity")
class JsonValidityMetric(BaseMetric):

    def score_batch(self, generated, references, samples):

        results = []
        for gen, sample in zip(generated, samples):
            gen_clean = re.sub(r"```(?:json)?\n?", "", gen).strip()

            parseable = 0.0
            schema_valid = 0.0
            schema_error = 0.0 

            raw_schema = sample.get("schema")
            try:
                schema = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
            except json.JSONDecodeError:
                schema = None
                schema_error = 1.0

            try:
                gen_obj = json.loads(gen_clean)
                parseable = 1.0
            except json.JSONDecodeError:
                gen_obj = None

            if parseable and schema is not None:
                try:
                    jsonschema.validate(gen_obj, schema)
                    schema_valid = 1.0
                except jsonschema.ValidationError:
                    schema_valid = 0.0
                except Exception:
                    schema_error = 1.0

            results.append({
                "primary": schema_valid,
                "parseable": parseable,
                "schema_valid": schema_valid,
                "schema_error": schema_error,
            })
        return results