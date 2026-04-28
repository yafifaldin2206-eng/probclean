# probclean

> A lightweight, drop-in library for correcting categorical typos in tabular data with confidence scores and explanations.

## The problem

You have a CSV. The `customer_name` column has `"Alicee"`, `"J0hn"`, `"Bobb"`. You know these are typos. You fix them manually. Every time. This library does it for you.

## Install

```bash
pip install probclean
```

## Quickstart

```python
import pandas as pd
from probclean import TypoCleaner

df = pd.read_csv("orders.csv")

cleaner = TypoCleaner(column="customer_name")
df_clean = cleaner.fit_transform(df)

cleaner.explain_changes()
```

## Output

```
original | corrected | confidence | reason
---------|-----------|------------|-------
J0hn     | John      | 0.89       | edit_distance=1, high_frequency_candidate
Alicee   | Alice     | 0.94       | edit_distance=1, high_frequency_candidate
Bobb     | Bob       | 0.81       | edit_distance=1, high_frequency_candidate
```

## What it solves

- ✅ Categorical typos in a single column
- ✅ Confidence scores per correction
- ✅ Human-readable explanations
- ✅ Drop-in pandas workflow

## What it does NOT solve

- ❌ Numerical outliers
- ❌ Missing value imputation
- ❌ Cross-column inference
- ❌ Product codes / IDs (disable per column)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `column` | required | Column name to clean |
| `top_k` | 20 | Number of canonical (clean) candidates to keep |
| `max_edit_distance` | 2 | Max edit distance to consider a value suspicious |
| `alpha` | 0.7 | Weight of similarity vs frequency prior |
| `min_confidence` | 0.5 | Minimum confidence to apply a correction |
| `min_freq` | 2 | Minimum frequency to be considered canonical |

## Honest limitations

- Rare-but-correct values (e.g. unusual names) may be incorrectly flagged → use `min_confidence`
- Ambiguous typos (`"Jon"` → `"John"` or `"Joan"`) → highest score wins, alternatives exposed
- Designed for human-readable categorical columns, not IDs or codes

## Benchmarks

On synthetic corruption of 2,000 rows (5% noise injected, 20 canonical names):

| Method | Accuracy | Precision | Coverage | Speed |
|--------|----------|-----------|----------|-------|
| Edit distance only | 100% | 100% | 100% | ~270ms |
| **probclean** | **96.3%** | **100%** | **96.3%** | **~18ms** |

**Key finding:** On a clean 20-name dataset, both methods achieve near-perfect precision. probclean is **15× faster** because it builds a candidate set up front rather than comparing every value against every other. The accuracy gap (3.7pp) reflects a small number of typos that fall below probclean's confidence threshold and are left unchanged — a conservative, safe-by-default behavior.

In messier real-world data (ambiguous canonicals, domain-specific values), probclean's frequency prior provides meaningful signal that pure edit distance lacks.

Run the benchmark yourself:
```bash
python benchmarks/run_benchmark.py
```

## License

MIT
