"""

Scores each candidate correction for a suspicious (dirty) value.

Algorithm:
    score(candidate | dirty) = α * similarity(dirty, candidate)
                             + (1 - α) * log_freq_prior(candidate)

Where:
    - similarity  = normalized Levenshtein in [0, 1]
    - log_freq    = log(freq_prior) normalized to [0, 1] across candidates
    - α           = weight controlling similarity vs frequency trade-off

Confidence:
    confidence = score_best / sum(all_scores)

This gives a soft measure of how dominant the top candidate is.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from probclean.edit_distance import levenshtein, normalized_similarity

logger = logging.getLogger(__name__)


@dataclass
class CorrectionCandidate:
    """Represents a scored correction candidate for a dirty value."""

    original: str
    candidate: str
    edit_distance: int
    similarity: float
    freq_prior: float
    score: float
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    
@dataclass
class CorrectionResult:
    """
    Final correction decision for a single dirty value.

    Attributes:
        original:     The raw (dirty) value from the data.
        corrected:    The best candidate, or original if no correction applied.
        confidence:   Score confidence in [0, 1].
        changed:      Whether a correction was applied.
        alternatives: Other candidates considered (sorted by score desc).
        reasons:      Human-readable explanation strings.
    """

    original: str
    corrected: str
    confidence: float
    changed: bool
    alternatives: List[CorrectionCandidate] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def _normalize_log_priors(priors: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize log-transformed frequency priors to [0, 1].

    Uses log(prior) to dampen the dominance of very frequent values,
    then min-max normalizes across the candidate set.
    """
    if not priors:
        return {}

    log_vals = {k: math.log(v + 1e-9) for k, v in priors.items()}
    min_v = min(log_vals.values())
    max_v = max(log_vals.values())

    if max_v == min_v:
        return {k: 1.0 for k in log_vals}

    return {k: (v - min_v) / (max_v - min_v) for k, v in log_vals.items()}

def score_candidates(
    dirty_value: str,
    canonical: Dict[str, int],
    freq_priors: Dict[str, float],
    alpha: float = 0.7,
    max_edit_distance: int = 2,
) -> List[CorrectionCandidate]:
    """
    Score all canonical candidates for a given dirty value.

    Only considers candidates within max_edit_distance.

    Args:
        dirty_value:       The suspicious value to correct.
        canonical:         Dict of canonical value → raw frequency.
        freq_priors:       Dict of canonical value → normalized frequency prior.
        alpha:             Similarity weight (1-alpha goes to frequency prior).
        max_edit_distance: Maximum edit distance to consider a candidate.

    Returns:
        List of CorrectionCandidate, sorted by score descending.
        Empty list if no candidates within max_edit_distance.
    """
    normalized_priors = _normalize_log_priors(freq_priors)
    candidates = []

    for canon_val, freq in canonical.items():
        dist = levenshtein(dirty_value, canon_val)
        if dist > max_edit_distance:
            continue

        sim = normalized_similarity(dirty_value, canon_val)
        norm_prior = normalized_priors.get(canon_val, 0.0)
        score = alpha * sim + (1 - alpha) * norm_prior

        reasons = _build_reasons(dist, sim, freq, canonical)

        candidates.append(
            CorrectionCandidate(
                original=dirty_value,
                candidate=canon_val,
                edit_distance=dist,
                similarity=sim,
                freq_prior=freq_priors.get(canon_val, 0.0),
                score=score,
                reasons=reasons,
            )
        )

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Assign confidence to each (relative to best)
    _assign_confidence(candidates)

    return candidates

def _build_reasons(
    dist: int,
    sim: float,
    freq: int,
    canonical: Dict[str, int],
) -> List[str]:
    """Build human-readable reason strings for a candidate."""
    reasons = []

    reasons.append(f"edit_distance={dist}")

    if sim >= 0.9:
        reasons.append("very_high_similarity")
    elif sim >= 0.75:
        reasons.append("high_similarity")
    else:
        reasons.append("moderate_similarity")

    max_freq = max(canonical.values()) if canonical else 1
    freq_pct = freq / max_freq
    if freq_pct >= 0.5:
        reasons.append("high_frequency_candidate")
    elif freq_pct >= 0.1:
        reasons.append("moderate_frequency_candidate")
    else:
        reasons.append("low_frequency_candidate")

    return reasons

def _assign_confidence(candidates: List[CorrectionCandidate]) -> None:
    """
    Assign confidence scores in-place.

    confidence_i = score_i / sum(all_scores)

    The top candidate's confidence reflects how dominant it is
    over all other candidates.
    """
    total = sum(c.score for c in candidates)
    if total == 0:
        for c in candidates:
            c.confidence = 0.0
        return
    for c in candidates:
        c.confidence = c.score / total


