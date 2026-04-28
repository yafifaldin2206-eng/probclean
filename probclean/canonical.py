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
