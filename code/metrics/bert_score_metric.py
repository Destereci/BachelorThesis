from metrics.base_metric import BaseMetric, register_metric




@register_metric("bertscore")
class BERTScoreMetric(BaseMetric):

    def __init__(self):
        self._bert_scorer = None
        self._rouge_scorer = None

    def _load(self):
        if self._bert_scorer is None:
            from bert_score import BERTScorer
            from rouge_score import rouge_scorer
            self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            self._rouge_scorer = rouge_scorer.RougeScorer(rouge_types=["rougeL"], use_stemmer=True)

    def score_batch(self, generated, references):
        self._load()
        P, R, F1 = self._bert_scorer.score(generated, references)
        results = []

        for i, (gen, ref) in enumerate(zip(generated, references)):
            rouge_score = self._rouge_scorer.score(ref, gen)
            results.append({
                "primary":   float(F1[i]),          # BERTScore F1
                "bertscore_f1": float(F1[i]),
                "bertscore_p":  float(P[i]),
                "bertscore_r":  float(R[i]),
                "rouge_l":      rouge_score["rougeL"].fmeasure,
            })
        return results