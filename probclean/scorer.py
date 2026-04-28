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

