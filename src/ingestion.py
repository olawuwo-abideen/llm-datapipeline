"""Stage 1: Data ingestion.

Loads CSV or JSON into a pandas DataFrame. Raises a clear error for
unsupported formats so failures happen early, at the pipeline boundary.
"""

from pathlib import Path

import pandas as pd


class DataIngestion:

    SUPPORTED = {".csv", ".json"}

    def load(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        extension = path.suffix.lower()

        if extension == ".csv":
            return pd.read_csv(path)

        if extension == ".json":
            return pd.read_json(path)

        raise ValueError(
            f"Unsupported file format '{extension}'. "
            f"Supported formats: {sorted(self.SUPPORTED)}"
        )
