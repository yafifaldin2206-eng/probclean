"""
canonical.py

Builds the canonical (clean) value set from a pandas Series.

Strategy:
- Rank values by frequency descending
- Keep top-K values with frequency >= min_freq
- These become the "ground truth" candidates for correction
"""
