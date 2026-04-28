"""
probclean — Lightweight categorical typo correction for tabular data.

Quick start:
    >>> import pandas as pd
    >>> from probclean import TypoCleaner
    >>> df = pd.DataFrame({"name": ["John"]*50 + ["Alice"]*40 + ["J0hn"]*3})
    >>> cleaner = TypoCleaner(column="name")
    >>> df_clean = cleaner.fit_transform(df)
    >>> cleaner.explain_changes()
"""

from probclean.cleaner import TypoCleaner

__version__ = "0.1.0"
__all__ = ["TypoCleaner"]
