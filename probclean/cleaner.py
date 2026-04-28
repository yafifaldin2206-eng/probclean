"""
cleaner.py

Public API: TypoCleaner

The single class users interact with. Wraps all internal modules
into a sklearn-style fit → transform pipeline.

Usage:
    cleaner = TypoCleaner(column="customer_name")
    df_clean = cleaner.fit_transform(df)
    report = cleaner.explain_changes()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from probclean.canonical import (
    build_canonical_set,
    frequency_prior,
    get_suspicious_values,
)
from probclean.scorer import (
    CorrectionResult,
    make_decision,
    score_candidates,
)

logger = logging.getLogger(__name__)


class TypoCleaner:
    """
    Drop-in categorical typo corrector for a single DataFrame column.

    Parameters
    ----------
    column : str
        Name of the column to clean.
    top_k : int, default 20
        Number of most-frequent values treated as canonical (clean) candidates.
    max_edit_distance : int, default 2
        Maximum Levenshtein distance to flag a value as suspicious.
        Set to 1 for conservative corrections only.
    alpha : float, default 0.7
        Weight given to string similarity vs frequency prior.
        alpha=1.0 → pure similarity.
        alpha=0.0 → pure frequency.
    min_confidence : float, default 0.5
        Minimum confidence to apply a correction.
        Values below threshold are left unchanged.
    min_freq : int, default 2
        Minimum occurrence count for a value to be canonical.
        Values appearing once are likely typos themselves.

    Examples
    --------
    >>> import pandas as pd
    >>> from probclean import TypoCleaner
    >>> df = pd.DataFrame({"name": ["John"]*50 + ["Alice"]*40 + ["J0hn"]*3})
    >>> cleaner = TypoCleaner(column="name")
    >>> df_clean = cleaner.fit_transform(df)
    >>> report = cleaner.explain_changes()
    """

    def __init__(
        self,
        column: str,
        top_k: int = 20,
        max_edit_distance: int = 2,
        alpha: float = 0.7,
        min_confidence: float = 0.5,
        min_freq: int = 2,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if max_edit_distance < 1:
            raise ValueError(f"max_edit_distance must be >= 1, got {max_edit_distance}")

        self.column = column
        self.top_k = top_k
        self.max_edit_distance = max_edit_distance
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.min_freq = min_freq

        # State set during fit()
        self._canonical: Optional[Dict[str, int]] = None
        self._freq_priors: Optional[Dict[str, float]] = None
        self._correction_map: Optional[Dict[str, CorrectionResult]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TypoCleaner":
        """
        Learn the canonical value set from df[column] and build correction map.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing self.column.

        Returns
        -------
        self
        """
        self._validate_df(df)

        series = df[self.column]

        # Step 1: Build canonical value set
        self._canonical = build_canonical_set(
            series, top_k=self.top_k, min_freq=self.min_freq
        )
        logger.info(
            "Built canonical set with %d values: %s",
            len(self._canonical),
            list(self._canonical.keys())[:5],
        )

        # Step 2: Compute frequency priors
        self._freq_priors = frequency_prior(self._canonical)

        # Step 3: Detect suspicious values
        suspicious = get_suspicious_values(
            series, self._canonical, self.max_edit_distance
        )
        logger.info("Detected %d suspicious values.", len(suspicious))

        # Step 4: Score and decide for each suspicious value
        self._correction_map = {}
        for dirty_val in suspicious:
            candidates = score_candidates(
                dirty_value=dirty_val,
                canonical=self._canonical,
                freq_priors=self._freq_priors,
                alpha=self.alpha,
                max_edit_distance=self.max_edit_distance,
            )
            result = make_decision(dirty_val, candidates, self.min_confidence)
            self._correction_map[dirty_val] = result

        self._is_fitted = True
        n_changed = sum(1 for r in self._correction_map.values() if r.changed)
        logger.info(
            "Fit complete: %d corrections will be applied.", n_changed
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned corrections to df[column].

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to correct. Does not modify in place.

        Returns
        -------
        pd.DataFrame
            Copy of df with corrected column.
        """
        self._check_fitted()
        self._validate_df(df)

        df_out = df.copy()
        df_out[self.column] = df_out[self.column].apply(self._apply_correction)
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on df and return corrected DataFrame in one step.

        Equivalent to cleaner.fit(df).transform(df).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_changes(self) -> pd.DataFrame:
        """
        Return a DataFrame summarizing all corrections made.

        Returns a pandas DataFrame (not a print statement) so the user
        can filter, sort, and inspect programmatically.

        Columns:
            original     - The dirty value
            corrected    - The applied correction
            confidence   - Score confidence in [0, 1]
            changed      - Whether a correction was applied
            edit_distance - Edit distance between original and corrected
            reasons      - Comma-separated reason strings
            alternatives - Other candidates considered

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()

        rows = []
        for dirty, result in self._correction_map.items():
            edit_dist = None
            if result.changed and result.alternatives is not None:
                from probclean.edit_distance import levenshtein
                edit_dist = levenshtein(result.original, result.corrected)
            elif result.changed:
                from probclean.edit_distance import levenshtein
                edit_dist = levenshtein(result.original, result.corrected)

            alt_names = [a.candidate for a in result.alternatives] if result.alternatives else []

            rows.append(
                {
                    "original": result.original,
                    "corrected": result.corrected,
                    "confidence": round(result.confidence, 4),
                    "changed": result.changed,
                    "edit_distance": edit_dist,
                    "reasons": ", ".join(result.reasons),
                    "alternatives": ", ".join(alt_names) if alt_names else "",
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "original", "corrected", "confidence",
                    "changed", "edit_distance", "reasons", "alternatives"
                ]
            )

        report = pd.DataFrame(rows)
        report = report.sort_values("confidence", ascending=False).reset_index(drop=True)
        return report

    def correction_map(self) -> Dict[str, str]:
        """
        Return a simple dict mapping dirty → corrected values.

        Only includes values where a correction was applied.

        Returns
        -------
        Dict[str, str]
        """
        self._check_fitted()
        return {
            dirty: result.corrected
            for dirty, result in self._correction_map.items()
            if result.changed
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_correction(self, value) -> str:
        """Apply correction for a single cell value."""
        if pd.isna(value):
            return value
        val_str = str(value).strip()
        if val_str in self._correction_map:
            result = self._correction_map[val_str]
            if result.changed:
                return result.corrected
        return val_str

    def _validate_df(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        if self.column not in df.columns:
            raise ValueError(
                f"Column '{self.column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "TypoCleaner is not fitted yet. Call fit() or fit_transform() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TypoCleaner("
            f"column='{self.column}', "
            f"top_k={self.top_k}, "
            f"alpha={self.alpha}, "
            f"min_confidence={self.min_confidence}, "
            f"status={status})"
        )
