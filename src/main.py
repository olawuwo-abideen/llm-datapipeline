"""Pipeline orchestrator.

STAGE ORDERING
--------------
    ingest -> validate -> PII anonymise -> toxicity/bias filter
           -> deduplicate -> clean/normalise -> format -> export
           -> metrics

PII and toxicity detection run on the ORIGINAL text (Presidio and
Detoxify rely on capitalisation and sentence structure); linguistic
normalisation runs last. Deduplication runs after filtering (fewer rows
to compare) but before cleaning, so near-duplicate detection sees the
anonymised natural text.

OUTPUT MODES
------------
    --mode instruction (default): Alpaca-style instruction/input/output
        JSON for LLM fine-tuning -> processed_data.json
    --mode mirror: output matches the input format (CSV in -> CSV out,
        JSON in -> JSON out) -> processed_data.<input extension>.
        Text is natural sentences by default in both modes; enable
        --lemmatize / --remove-stopwords only for classical NLP use.
        Only the text column is exported unless --keep-columns is set,
        because metadata columns may carry unsanitised PII.

Examples:
    python main.py --input data.csv
    python main.py --input data.csv --mode mirror
    python main.py --input data.csv --mode mirror --output clean.csv
    python main.py --input data.json --toxicity-threshold 0.7 --lemmatize --remove-stopwords
"""

import argparse
from pathlib import Path

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
    parser.add_argument("--output", default=None,
                        help="Output file. Defaults to processed_data.json "
                             "(instruction mode) or processed_data.<input "
                             "extension> (mirror mode)")
    parser.add_argument("--mode", choices=["instruction", "mirror"],
                        default="instruction",
                        help="instruction: Alpaca-style JSON for fine-tuning. "
                             "mirror: output matches the input format "
                             "(CSV in -> CSV out) with sanitised text")
    parser.add_argument("--keep-columns", action="store_true",
                        help="Mirror mode only: keep all input columns, not "
                             "just the sanitised text. WARNING: metadata "
                             "columns may contain unsanitised PII")
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
    parser.add_argument("--lemmatize", action="store_true",
                        help="Enable lemmatisation (opt-in; off by default "
                             "so output keeps natural sentences)")
    parser.add_argument("--remove-stopwords", action="store_true",
                        help="Enable stopword removal (opt-in; off by default)")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    elif args.mode == "mirror":
        output_path = "processed_data" + Path(args.input).suffix.lower()
    else:
        output_path = "processed_data.json"

    config = PipelineConfig(
        input_path=args.input,
        output_path=output_path,
        metrics_path=args.metrics,
        similarity_threshold=args.similarity_threshold,
        toxicity_threshold=args.toxicity_threshold,
        instruction_text=args.instruction,
        lemmatize=args.lemmatize,
        remove_stopwords=args.remove_stopwords,
    )
    config.mode = args.mode
    config.keep_columns = args.keep_columns
    return config


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

    # Stage 7 + 8: format and export according to the selected mode
    mode = getattr(config, "mode", "instruction")
    if mode == "mirror":
        saved_path = exporter.save_mirror(
            df, config.output_path,
            keep_columns=getattr(config, "keep_columns", False))
        print(f"Sanitised dataset (mirror mode) saved to: {saved_path}")
    else:
        formatted = formatter.format(df)
        saved_path = exporter.save_instruction(formatted, config.output_path)
        print(f"Instruction dataset saved to: {saved_path}")

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
