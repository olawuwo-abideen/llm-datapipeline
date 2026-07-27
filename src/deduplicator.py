"""Stage 5: Duplicate and near-duplicate removal.

TF-IDF + cosine similarity, as before, with three fixes:

1. Guard clauses for empty / single-row inputs (TfidfVectorizer raises
   on an empty corpus).
2. The O(n^2) Python double loop is replaced with a vectorised numpy
   pass over the upper triangle of the similarity matrix. The memory
   complexity is still O(n^2), which is acceptable at dissertation
   scale but should be noted as a limitation for very large corpora
   (MinHash / LSH would be the scalable alternative).
3. Exact duplicates are removed first with drop_duplicates(), which is
   cheap and shrinks the similarity computation.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Deduplicator:

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.exact_removed = 0
        self.near_removed = 0

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pass 1: exact duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        self.exact_removed = before - len(df)

        texts = df["text"].tolist()

        # Guard: similarity is undefined for fewer than two documents
        if len(texts) < 2:
            return df

        # Pass 2: near-duplicates
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Upper triangle only (j > i), vectorised
        upper = np.triu(similarity_matrix, k=1)
        _, duplicate_cols = np.where(upper > self.threshold)
        duplicate_indices = sorted(set(duplicate_cols.tolist()))

        self.near_removed = len(duplicate_indices)

        if duplicate_indices:
            df = df.drop(df.index[duplicate_indices]).reset_index(drop=True)

        return df
