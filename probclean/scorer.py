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
