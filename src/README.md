---
title: LLM Data Sanitisation Pipeline
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Data Sanitisation Pipeline for LLM Fine-Tuning

A modular preprocessing pipeline that prepares text datasets for large language model fine-tuning: cleaning, PII anonymisation, toxicity and bias filtering, deduplication, instruction formatting, and evaluation metrics.

## Pipeline order

```
ingest -> validate -> PII anonymise -> toxicity/bias filter
       -> deduplicate (exact + near) -> clean/normalise
       -> format -> export -> metrics
```

PII detection and toxicity scoring run on the original text because Presidio relies on capitalisation and sentence structure, and Detoxify was trained on natural sentences. Linguistic normalisation (lowercasing, lemmatisation, stopword removal) therefore runs last, and is switchable off entirely for fine-tuning tasks that need natural sentences.

## Usage

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python main.py --input ../datasets/data.csv --output processed.json

# Examples of configurability
python main.py --input data.json --toxicity-threshold 0.7 --no-lemmatize
python main.py --input data.csv --similarity-threshold 0.85 --instruction "Summarise the following text"
```

Outputs: the formatted instruction dataset (`processed.json`) and a metrics report (`metrics_report.json`) containing per-stage quantitative counts and qualitative before/after samples.

## Changes made in this revision (code review summary)

1. **Stage reordering (correctness).** Cleaning previously ran before PII and toxicity detection, degrading both. PII and toxicity now operate on original text; normalisation runs last.
2. **Exporter wired in.** `main.py` previously never called the exporter, so no artefact was produced. The output path is now a parameter.
3. **Metrics corrected and extended.** `duplicates_removed` previously conflated validation drops with duplicates. Each stage now reports its own count; exact and near duplicates are separated; a JSON report with qualitative before/after samples is written, supporting quantitative and qualitative evaluation.
4. **Bias coverage.** Filtering now checks Detoxify's `identity_attack`, `insult`, and `threat` scores as well as `toxicity`, addressing the "biased, toxic, or harmful content" objective. A per-attribute breakdown appears in the metrics.
5. **Toxic rows dropped, not replaced.** A "[TOXIC CONTENT REMOVED]" placeholder would itself become a training example; rows are excluded instead.
6. **Batched toxicity inference.** Detoxify is called on batches rather than row by row, a large speedup.
7. **Configurability.** An `argparse` CLI and a `PipelineConfig` dataclass replace hardcoded paths and thresholds, supporting the flexible, reusable framework objective.
8. **Robustness.** `ValueError` instead of bare `Exception`; guards for empty and single-row inputs in deduplication; non-string and empty-string handling in validation and cleaning; anonymisation placeholders (e.g. `[EMAIL]`) protected from lemmatisation; rows emptied by cleaning are dropped.
9. **Deduplication improvements.** Cheap exact-duplicate pass first, then a vectorised numpy pass over the similarity matrix instead of a Python double loop. The O(n^2) memory of the similarity matrix remains a documented limitation; MinHash/LSH is the scalable alternative for very large corpora.

## Known limitations (worth stating in the dissertation)

- Pairwise similarity is O(n^2) in memory; suitable at dissertation scale, not web scale.
- Detoxify and Presidio are English-focused; multilingual data is out of scope.
- The `output` field in formatted records is intentionally empty: producing target completions is a labelling task outside a sanitisation pipeline's scope.
- Threshold choices (0.90 similarity, 0.60 toxicity) are defaults; sensitivity analysis of these thresholds is a natural part of the evaluation chapter.
