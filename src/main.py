"""Pipeline orchestrator.

CORRECTED STAGE ORDERING
------------------------
The earlier version cleaned (lowercase + lemmatise) BEFORE PII and
toxicity detection, which degraded both: Presidio needs capitalisation
and sentence structure to find names/locations, and Detoxify was
trained on natural sentences. The corrected order is:

    ingest -> validate -> PII anonymise -> toxicity/bias filter
           -> deduplicate -> clean/normalise -> format -> export
           -> metrics

Deduplication runs after filtering (fewer rows to compare) but before
cleaning, so near-duplicate detection sees the anonymised natural text.

A command-line interface replaces the hardcoded paths and thresholds,
satisfying the objective of a flexible, reusable framework:

    python main.py --input ../datasets/data.csv --output processed.json
    python main.py --input data.csv --no-lemmatize --toxicity-threshold 0.7
"""

import argparse

from cleaner import Cleaner
from config import PipelineConfig
from deduplicator import Deduplicator
from exporter import Exporter
from formatter import Formatter
from ingestion import DataIngestion
from metrics import Metrics
from pii_detector import PIIDetector
from toxicity_filter import ToxicityFilter
from validator import Validator


def parse_args() -> PipelineConfig:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Data sanitisation pipeline for LLM fine-tuning datasets"
    )
    parser.add_argument("--input", default=defaults.input_path,
                        help="Input CSV or JSON file")
    parser.add_argument("--output", default=defaults.output_path,
                        help="Output JSON file for the formatted dataset")
    parser.add_argument("--metrics", default=defaults.metrics_path,
                        help="Output JSON file for the metrics report")
    parser.add_argument("--similarity-threshold", type=float,
                        default=defaults.similarity_threshold,
                        help="Cosine similarity threshold for near-duplicates")
    parser.add_argument("--toxicity-threshold", type=float,
                        default=defaults.toxicity_threshold,
                        help="Detoxify score threshold for removal")
    parser.add_argument("--instruction", default=defaults.instruction_text,
                        help="Instruction text for the formatted records")
    parser.add_argument("--no-lemmatize", action="store_true",
                        help="Disable lemmatisation (keep natural sentences)")
    parser.add_argument("--keep-stopwords", action="store_true",
                        help="Disable stopword removal")
    args = parser.parse_args()

    return PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        metrics_path=args.metrics,
        similarity_threshold=args.similarity_threshold,
        toxicity_threshold=args.toxicity_threshold,
        instruction_text=args.instruction,
        lemmatize=not args.no_lemmatize,
        remove_stopwords=not args.keep_stopwords,
    )


def run(config: PipelineConfig) -> None:
    ingestion = DataIngestion()
    validator = Validator()
    pii = PIIDetector()
    toxic = ToxicityFilter(
        threshold=config.toxicity_threshold,
        attributes=config.filtered_attributes,
    )
    dedup = Deduplicator(threshold=config.similarity_threshold)
    cleaner = Cleaner(
        lemmatize=config.lemmatize,
        remove_stopwords=config.remove_stopwords,
    )
    formatter = Formatter(instruction=config.instruction_text)
    exporter = Exporter()
    metrics = Metrics()

    # Stage 1: ingest
    df = ingestion.load(config.input_path)
    before = df.copy()

    # Stage 2: validate
    df = validator.validate(df)

    # Stage 3: PII anonymisation (on original text)
    df["text"] = df["text"].apply(pii.anonymize)

    # Stage 4: toxicity / bias filtering (on original text, batched)
    df = toxic.filter_dataframe(df)

    # Stage 5: exact + near-duplicate removal
    df = dedup.remove_duplicates(df)

    # Stage 6: cleaning and normalisation (last, by design, batched)
    df["text"] = cleaner.clean_series(df["text"].tolist())

    # Drop rows that became empty after cleaning
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    after = df.copy()

    # Stage 7: instruction formatting
    formatted = formatter.format(df)

    # Stage 8: export
    saved_path = exporter.save(formatted, config.output_path)
    print(f"Formatted dataset saved to: {saved_path}")

    # Stage 9: metrics
    metrics.generate(
        before_df=before,
        after_df=after,
        validation_dropped=validator.rows_dropped,
        exact_duplicates=dedup.exact_removed,
        near_duplicates=dedup.near_removed,
        pii_found=pii.pii_count,
        pii_rows=pii.rows_affected,
        toxic_flagged=toxic.flagged,
        toxic_by_attribute=toxic.flagged_by_attribute,
        report_path=config.metrics_path,
    )


if __name__ == "__main__":
    run(parse_args())
