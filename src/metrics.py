"""Stage 9: Evaluation metrics.

Fixes and extensions over the earlier version:

1. Per-stage attribution. Previously duplicates_removed was computed as
   before_size - after_size, which silently included rows dropped by
   validation and toxicity filtering. Each stage now reports its own
   count, so the numbers are correct and auditable.

2. Persistent report. Metrics are written to JSON as well as printed,
   so the dissertation's evaluation chapter has a reproducible record.

3. Qualitative samples. A small set of before/after text pairs is
   included, supporting the objective of evaluating the pipeline with
   qualitative as well as quantitative measures.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class Metrics:

    def generate(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        validation_dropped: int,
        exact_duplicates: int,
        near_duplicates: int,
        pii_found: int,
        pii_rows: int,
        toxic_flagged: int,
        toxic_by_attribute: dict,
        sample_size: int = 5,
        report_path: str = "metrics_report.json",
    ) -> dict:

        before_size = len(before_df)
        after_size = len(after_df)

        reduction_percent = (
            ((before_size - after_size) / before_size) * 100
            if before_size else 0.0
        )

        # Qualitative before/after samples (rows surviving to the end)
        n = min(sample_size, after_size, before_size)
        samples = [
            {
                "before": str(before_df["text"].iloc[i])[:300],
                "after": str(after_df["text"].iloc[i])[:300],
            }
            for i in range(n)
        ]

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "quantitative": {
                "original_rows": before_size,
                "final_rows": after_size,
                "reduction_percent": round(reduction_percent, 2),
                "dropped_by_validation": validation_dropped,
                "exact_duplicates_removed": exact_duplicates,
                "near_duplicates_removed": near_duplicates,
                "pii_entities_anonymised": pii_found,
                "rows_containing_pii": pii_rows,
                "rows_removed_as_toxic_or_biased": toxic_flagged,
                "toxicity_breakdown_by_attribute": toxic_by_attribute,
            },
            "qualitative_samples": samples,
        }

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        self._print_summary(report["quantitative"], report_path)

        return report

    @staticmethod
    def _print_summary(q: dict, report_path: str) -> None:
        print("\n===== PIPELINE METRICS =====")
        print(f"Original dataset:          {q['original_rows']}")
        print(f"Final dataset:             {q['final_rows']}")
        print(f"Reduction %:               {q['reduction_percent']}")
        print(f"Dropped by validation:     {q['dropped_by_validation']}")
        print(f"Exact duplicates removed:  {q['exact_duplicates_removed']}")
        print(f"Near duplicates removed:   {q['near_duplicates_removed']}")
        print(f"PII entities anonymised:   {q['pii_entities_anonymised']}")
        print(f"Rows containing PII:       {q['rows_containing_pii']}")
        print(f"Toxic/biased rows removed: {q['rows_removed_as_toxic_or_biased']}")
        print(f"Toxicity breakdown:        {q['toxicity_breakdown_by_attribute']}")
        print(f"Full report saved to:      {report_path}")
