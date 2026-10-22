"""Community-specific vocabulary via weighted log-odds (Monroe et al. 2008).

Uses the informative Dirichlet prior variant: the prior for each term is the
term's frequency in the pooled corpus scaled to ``alpha0`` pseudo-counts. The
output is a ranked list of terms by z-score; no word lists are hand-written.
Results are corpus-level aggregates and are used only to ablate text features.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens matching the TF-IDF default token pattern."""
    return TOKEN.findall(text.lower())


def count_terms(docs: Iterable[str]) -> Counter:
    """Total unigram counts over documents."""
    counts: Counter = Counter()
    for doc in docs:
        counts.update(tokenize(doc))
    return counts


def log_odds_informative_dirichlet(
    target_counts: Counter,
    contrast_counts: Counter,
    alpha0: float = 500.0,
    min_count: int = 5,
) -> list[tuple[str, float]]:
    """Rank terms by z-scored log-odds toward the target corpus.

    Args:
        target_counts: Term counts in the target community (train only).
        contrast_counts: Term counts in the contrast community (train only).
        alpha0: Total prior pseudo-count mass.
        min_count: Minimum pooled count for a term to be ranked.

    Returns:
        ``[(term, z)]`` sorted by descending z (most target-specific first).
    """
    vocab = [w for w in set(target_counts) | set(contrast_counts)
             if target_counts[w] + contrast_counts[w] >= min_count]
    if not vocab:
        return []
    y1 = np.array([target_counts[w] for w in vocab], dtype=float)
    y2 = np.array([contrast_counts[w] for w in vocab], dtype=float)
    pooled = y1 + y2
    a = alpha0 * pooled / pooled.sum()
    n1, n2 = y1.sum(), y2.sum()
    delta = np.log((y1 + a) / (n1 + alpha0 - y1 - a)) - np.log((y2 + a) / (n2 + alpha0 - y2 - a))
    var = 1.0 / (y1 + a) + 1.0 / (y2 + a)
    z = delta / np.sqrt(var)
    order = np.argsort(-z)
    return [(vocab[i], float(z[i])) for i in order]


def top_terms(ranked: Sequence[tuple[str, float]], k: int) -> set[str]:
    """The ``k`` most target-specific terms."""
    return {w for w, _ in ranked[:k]}


def strip_terms(docs: Iterable[str], terms: set[str]) -> list[str]:
    """Remove the given tokens from each document (and so from all n-grams)."""
    return [" ".join(t for t in tokenize(doc) if t not in terms) for doc in docs]
