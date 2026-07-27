"""Stage 2: Validation.

Checks schema and removes null / empty / non-string rows. Tracks how
many rows were dropped so metrics can distinguish validation drops from
deduplication (a bug in the earlier version conflated the two).
"""

import pandas as pd


class Validator:

    REQUIRED_COLUMNS = ["text"]

    def __init__(self):
        self.rows_dropped = 0

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.REQUIRED_COLUMNS:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        before = len(df)

        # Drop nulls
        df = df.dropna(subset=["text"])

        # Drop non-strings and empty / whitespace-only strings
        df = df[df["text"].apply(lambda t: isinstance(t, str) and t.strip() != "")]

        self.rows_dropped = before - len(df)

        return df.reset_index(drop=True)
