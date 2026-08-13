"""
Hinglish: Standard detection-evaluation metrics - TP/FP/FN counting se
precision, recall, F1 nikalte hain. Ye "span-level" matching use karta hai:
ek predicted span ground-truth span se match karta hai agar unka category
same ho aur character-offsets overlap karein (exact match nahi, overlap
kaafi hai - real-world span boundaries thoda vary kar sakte hain).
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LabeledSpan:
    start: int
    end: int
    category: str


@dataclass
class CategoryMetrics:
    category: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0 if self.fn == 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0 if self.fp == 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _spans_overlap(a: LabeledSpan, b: LabeledSpan) -> bool:
    return a.category == b.category and a.start < b.end and a.end > b.start


def compute_metrics(
    predicted: List[LabeledSpan], ground_truth: List[LabeledSpan]
) -> List[CategoryMetrics]:
    """
    Hinglish: Predicted spans ko ground-truth ke saath match karke
    per-category TP/FP/FN nikalta hai.

    Matching greedy hai: har ground-truth span ko zyada se zyada ek
    predicted span se match karte hain (aur vice versa), taaki duplicate
    counting na ho.
    """
    categories = sorted(set(s.category for s in predicted) | set(s.category for s in ground_truth))
    results = []
    for category in categories:
        preds = [s for s in predicted if s.category == category]
        gts = [s for s in ground_truth if s.category == category]
        matched_gt = set()
        matched_pred = set()
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                if gi in matched_gt:
                    continue
                if _spans_overlap(p, g):
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break
        tp = len(matched_gt)
        fp = len(preds) - len(matched_pred)
        fn = len(gts) - len(matched_gt)
        results.append(CategoryMetrics(category=category, tp=tp, fp=fp, fn=fn))
    return results


def overall_metrics(per_category: List[CategoryMetrics]) -> CategoryMetrics:
    """Hinglish: Sabhi categories ke TP/FP/FN ko sum karke overall (micro-average) metrics."""
    tp = sum(m.tp for m in per_category)
    fp = sum(m.fp for m in per_category)
    fn = sum(m.fn for m in per_category)
    return CategoryMetrics(category="OVERALL", tp=tp, fp=fp, fn=fn)
