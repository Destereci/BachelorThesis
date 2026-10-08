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
        empty_flags = [not g.strip() for g in generated]
        safe_gen = [g.strip() if g.strip() else "." for g in generated]
        P, R, F1 = self._bert_scorer.score(safe_gen, references)
        results = []

        for i, (gen, ref) in enumerate(zip(safe_gen, references)):
            rouge_score = self._rouge_scorer.score(ref, gen)
            results.append({
                "primary":   float(F1[i]),
                "bertscore_f1": float(F1[i]),
                "bertscore_p":  float(P[i]),
                "bertscore_r":  float(R[i]),
                "rouge_l":      rouge_score["rougeL"].fmeasure,
                "empty": empty_flags[i]
            })
        return results