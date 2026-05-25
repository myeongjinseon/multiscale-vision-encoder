"""
Evaluation Metrics for Captioning and Grounding.

Captioning metrics:
- CIDEr: Consensus-based, weights n-grams by TF-IDF (primary metric)
- BLEU-4: Precision of 4-gram overlap with references
- METEOR: Harmonic mean of precision/recall with stemming/synonyms

Grounding metrics:
- IoU: Intersection over Union between predicted and target boxes
- Acc@0.5: Fraction of predictions with IoU ≥ 0.5 (primary metric)

Implementations are self-contained for reproducibility.
For paper submission, also validate against official COCO eval tools.
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import math


# ============================================================
#  CAPTIONING METRICS
# ============================================================

class BLEUScorer:
    """
    BLEU-N score computation.
    
    Measures n-gram precision of candidate against references,
    with brevity penalty for short candidates.
    """

    @staticmethod
    def compute_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def score(
        candidates: List[List[str]],
        references: List[List[List[str]]],
        max_n: int = 4,
    ) -> Dict[str, float]:
        """
        Compute BLEU-1 through BLEU-N.
        
        Args:
            candidates: List of tokenized candidate captions
            references: List of lists of tokenized reference captions
            max_n: Maximum n-gram order
            
        Returns:
            Dict with 'BLEU-1', 'BLEU-2', ..., 'BLEU-N' scores
        """
        assert len(candidates) == len(references)

        # Collect clipped counts for each n
        precisions = []
        total_cand_len = 0
        total_ref_len = 0

        for n in range(1, max_n + 1):
            clipped_count = 0
            total_count = 0

            for cand, refs in zip(candidates, references):
                cand_ngrams = BLEUScorer.compute_ngrams(cand, n)

                # Maximum reference count for each n-gram
                max_ref_counts = Counter()
                for ref in refs:
                    ref_ngrams = BLEUScorer.compute_ngrams(ref, n)
                    for ng, count in ref_ngrams.items():
                        max_ref_counts[ng] = max(max_ref_counts[ng], count)

                # Clip candidate counts
                for ng, count in cand_ngrams.items():
                    clipped_count += min(count, max_ref_counts.get(ng, 0))
                    total_count += count

            precision = clipped_count / max(total_count, 1)
            precisions.append(precision)

        # Brevity penalty
        for cand, refs in zip(candidates, references):
            total_cand_len += len(cand)
            # Use closest reference length
            ref_lens = [len(r) for r in refs]
            closest = min(ref_lens, key=lambda r: (abs(r - len(cand)), r))
            total_ref_len += closest

        if total_cand_len > total_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - total_ref_len / max(total_cand_len, 1))

        # Compute BLEU scores
        results = {}
        for n in range(1, max_n + 1):
            log_avg = sum(math.log(max(p, 1e-10)) for p in precisions[:n]) / n
            results[f"BLEU-{n}"] = bp * math.exp(log_avg)

        return results


class CIDErScorer:
    """
    CIDEr-D score computation.
    
    Consensus-based metric that uses TF-IDF weighting of n-grams.
    Higher weight for n-grams that are common across references for
    an image but rare across the dataset (discriminative descriptions).
    """

    def __init__(self, n: int = 4, sigma: float = 6.0):
        self.n = n
        self.sigma = sigma

    def _compute_doc_freq(
        self, references: List[List[List[str]]]
    ) -> Counter:
        """Compute document frequency of n-grams across all reference sets."""
        df = Counter()
        for refs in references:
            # Count each n-gram once per image (union over references)
            seen = set()
            for ref in refs:
                for k in range(1, self.n + 1):
                    ngrams = set(tuple(ref[i:i+k]) for i in range(len(ref) - k + 1))
                    seen.update(ngrams)
            for ng in seen:
                df[ng] += 1
        return df

    def _compute_tfidf(
        self, tokens: List[str], df: Counter, num_docs: int
    ) -> Dict[Tuple, float]:
        """Compute TF-IDF vector for a single caption."""
        vec = defaultdict(float)
        length = len(tokens)

        for k in range(1, self.n + 1):
            for i in range(length - k + 1):
                ng = tuple(tokens[i:i+k])
                tf = 1.0 / max(length - k + 1, 1)
                idf = math.log(max(1.0, num_docs) / max(1.0, df.get(ng, 0)))
                vec[ng] += tf * idf

        return vec

    def _sim(self, vec1: Dict, vec2: Dict) -> float:
        """Cosine similarity between TF-IDF vectors."""
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values())) or 1e-10
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values())) or 1e-10

        dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
        return dot / (norm1 * norm2)

    def score(
        self,
        candidates: List[List[str]],
        references: List[List[List[str]]],
    ) -> float:
        """
        Compute corpus-level CIDEr-D score.
        
        Args:
            candidates: List of tokenized candidate captions
            references: List of lists of tokenized reference captions
            
        Returns:
            CIDEr-D score (typically 0-2, higher is better)
        """
        num_docs = len(references)
        df = self._compute_doc_freq(references)

        scores = []
        for cand, refs in zip(candidates, references):
            cand_vec = self._compute_tfidf(cand, df, num_docs)

            ref_scores = []
            for ref in refs:
                ref_vec = self._compute_tfidf(ref, df, num_docs)
                ref_scores.append(self._sim(cand_vec, ref_vec))

            # Average similarity across references
            scores.append(np.mean(ref_scores) if ref_scores else 0.0)

        return 10.0 * np.mean(scores)  # Scale by 10 (CIDEr convention)


class METEORScorer:
    """
    Simplified METEOR score.
    
    Computes unigram precision and recall with harmonic mean.
    Full METEOR also considers stemming, synonyms, and chunk penalty;
    this is a simplified version for quick evaluation.
    """

    @staticmethod
    def score(
        candidates: List[List[str]],
        references: List[List[List[str]]],
        alpha: float = 0.9,
    ) -> float:
        """
        Compute METEOR score.
        
        Args:
            candidates: Tokenized candidates
            references: Tokenized references (multiple per image)
            alpha: Weight parameter (higher = more weight on recall)
            
        Returns:
            METEOR score (0-1)
        """
        scores = []

        for cand, refs in zip(candidates, references):
            best_score = 0.0
            cand_set = Counter(cand)

            for ref in refs:
                ref_set = Counter(ref)

                # Unigram matches
                matches = sum((cand_set & ref_set).values())
                precision = matches / max(len(cand), 1)
                recall = matches / max(len(ref), 1)

                if precision + recall == 0:
                    f_score = 0.0
                else:
                    f_score = (precision * recall) / (
                        alpha * precision + (1 - alpha) * recall
                    )

                # Chunk penalty (simplified: count contiguous match chunks)
                chunks = METEORScorer._count_chunks(cand, ref)
                penalty = 0.5 * (chunks / max(matches, 1)) ** 3 if matches > 0 else 0

                score = f_score * (1 - penalty)
                best_score = max(best_score, score)

            scores.append(best_score)

        return np.mean(scores)

    @staticmethod
    def _count_chunks(cand: List[str], ref: List[str]) -> int:
        """Count the number of contiguous matching chunks."""
        ref_positions = defaultdict(list)
        for i, w in enumerate(ref):
            ref_positions[w].append(i)

        matched_positions = []
        for w in cand:
            if w in ref_positions and ref_positions[w]:
                matched_positions.append(ref_positions[w].pop(0))

        if not matched_positions:
            return 0

        chunks = 1
        for i in range(1, len(matched_positions)):
            if matched_positions[i] != matched_positions[i-1] + 1:
                chunks += 1

        return chunks


# ============================================================
#  GROUNDING METRICS
# ============================================================

def compute_iou(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU between predicted and target boxes.
    
    Args:
        pred: (B, 4) predicted boxes [cx, cy, w, h] normalized
        target: (B, 4) target boxes [cx, cy, w, h] normalized
        
    Returns:
        iou: (B,) IoU values
    """
    # Convert center format to corners
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = pred[:, 2] * pred[:, 3]
    tgt_area = target[:, 2] * target[:, 3]
    union_area = pred_area + tgt_area - inter_area

    return inter_area / union_area.clamp(min=1e-6)


def accuracy_at_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Acc@IoU (fraction of samples with IoU ≥ threshold).
    
    Args:
        pred: (B, 4) predicted boxes
        target: (B, 4) target boxes
        threshold: IoU threshold (default 0.5)
        
    Returns:
        accuracy: float in [0, 1]
    """
    iou = compute_iou(pred, target)
    return (iou >= threshold).float().mean().item()


# ============================================================
#  UNIFIED EVALUATOR
# ============================================================

class CaptioningEvaluator:
    """Unified evaluator for image captioning metrics."""

    def __init__(self):
        self.bleu_scorer = BLEUScorer()
        self.cider_scorer = CIDErScorer()
        self.meteor_scorer = METEORScorer()

    def evaluate(
        self,
        candidates: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate captions using all metrics.
        
        Args:
            candidates: List of candidate caption strings
            references: List of lists of reference caption strings
            
        Returns:
            Dict with all metric scores
        """
        # Tokenize (simple whitespace + lowercase)
        cand_tokens = [c.lower().split() for c in candidates]
        ref_tokens = [[r.lower().split() for r in refs] for refs in references]

        results = {}

        # BLEU
        bleu = self.bleu_scorer.score(cand_tokens, ref_tokens)
        results.update(bleu)

        # CIDEr
        results["CIDEr"] = self.cider_scorer.score(cand_tokens, ref_tokens)

        # METEOR
        results["METEOR"] = self.meteor_scorer.score(cand_tokens, ref_tokens)

        return results


class GroundingEvaluator:
    """Unified evaluator for visual grounding metrics."""

    @staticmethod
    def evaluate(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        thresholds: List[float] = [0.5, 0.75],
    ) -> Dict[str, float]:
        """
        Evaluate grounding predictions.
        
        Args:
            predictions: (N, 4) predicted boxes
            targets: (N, 4) target boxes
            thresholds: IoU thresholds for accuracy
            
        Returns:
            Dict with IoU and accuracy metrics
        """
        iou = compute_iou(predictions, targets)

        results = {
            "mean_IoU": iou.mean().item(),
            "median_IoU": iou.median().item(),
        }

        for t in thresholds:
            results[f"Acc@{t}"] = (iou >= t).float().mean().item()

        return results
