from metrics.base_metric import BaseMetric, register_metric
from bert_score import BERTScorer
from rouge_score import rouge_scorer



@register_metric("bertscore")
class BERTScoreMetric(BaseMetric):

    def __init__(self):
        self._bert_scorer = None
        self._rouge_scorer = None

    def _load(self):
        if self._bert_scorer is None:
            self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            self._rouge_scorer = rouge_scorer.RougeScorer(rouge_types=["rougeL"], use_stemmer=True)

    def score_batch(self, generated, references, samples):
        self._load()
        results = [None] * len(generated)
        nonempty_indices = [
            i for i, gen in enumerate(generated) if gen.strip()
        ]
        if nonempty_indices:
            nonempty_gen = [
                generated[i].strip() for i in nonempty_indices]
            nonempty_refs = [
                references[i] for i in nonempty_indices]
            P, R, F1 = self._bert_scorer.score(
                nonempty_gen, nonempty_refs)

            for j, original_i in enumerate(nonempty_indices):
                rouge = self._rouge_scorer.score(
                    nonempty_refs[j],
                    nonempty_gen[j]
                )
                results[original_i] = {
                    "primary": float(F1[j]),
                    "bertscore_f1": float(F1[j]),
                    "bertscore_p": float(P[j]),
                    "bertscore_r": float(R[j]),
                    "rouge_l": rouge["rougeL"].fmeasure,
                    "empty": False
                }
        for i, gen in enumerate(generated):
            if not gen.strip():
                results[i] = {
                    "primary": None,
                    "bertscore_f1": None,
                    "bertscore_p": None,
                    "bertscore_r": None,
                    "rouge_l": None,
                    "empty": True
                }

        return results