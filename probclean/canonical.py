"""
canonical.py

Builds the canonical (clean) value set from a pandas Series.

Strategy:
- Rank values by frequency descending
- Keep top-K values with frequency >= min_freq
- These become the "ground truth" candidates for correction
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def build_canonical_set(
  series: pd.Series,
  top_k: int = 20,
  min_freq: int =2,
) -> Dict[str, int]:
  if series.empty:
        raise ValueError("Series is empty — cannot build canonical set.")

    # Drop nulls, cast to string
    clean = series.dropna().astype(str).str.strip()
    if clean.empty:
        raise ValueError("Series has no non-null values.")

    counts = clean.value_counts()

    # Filter by minimum frequency
    counts = counts[counts >= min_freq]
    if counts.empty:
        raise ValueError(
            f"No values appear >= {min_freq} times. "
            "Lower min_freq or provide more data."
        )

    # Take top-K
    top = counts.head(top_k)

    return top.to_dict()

def frequency_prior(canonical: Dict[str, int]) -> Dict[str, float]:
  total = sum(canonical.values())
    if total == 0:
        raise ValueError("Canonical set has zero total frequency.")
    return {v: freq / total for v, freq in canonical.items()}

def get_suspicious_values(
    series: pd.Series,
    canonical: Dict[str, int],
    max_edit_distance: int = 2,
) -> List[str]:
  from probclean.edit_distance import levenshtein

    canonical_keys = set(canonical.keys())
    unique_values = set(series.dropna().astype(str).str.strip().unique())

    suspicious = []
    for val in unique_values:
        val_freq = canonical.get(val, 0)
      for canon in canonical_keys:
            if canon == val:
                continue
            canon_freq = canonical.get(canon, 0)
            # Only flag val as suspicious if a higher-freq neighbor is close
            if canon_freq > val_freq and levenshtein(val, canon) <= max_edit_distance:
                suspicious.append(val)
                break

    return sorted(suspicious)
      
