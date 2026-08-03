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

A modular preprocessing pipeline that prepares text datasets for large language model fine-tuning: PII anonymisation, toxicity and implicit hate filtering, exact and near-duplicate removal, configurable cleaning, instruction formatting, and fully attributed evaluation metrics.

**MSc project artefact.** Author: Abideen Olawuwo (Student ID 25043520), UWE Bristol, UFCF9Y-60-M CSCT Masters Project.

**Live demo:** https://llm-datapipeline.streamlit.app/(reduced configuration, see Hosted vs local below)

## Pipeline order

```
ingest -> validate -> PII anonymise -> toxicity/bias filter
       -> deduplicate (exact + near) -> clean/normalise
       -> format -> export -> metrics
```

PII detection and toxicity scoring run on the original text because Presidio's NER relies on capitalisation and sentence structure, and the toxicity classifiers were trained on natural sentences. Linguistic normalisation (lowercasing, lemmatisation, stopword removal) therefore runs last and is off by default, since fine-tuning data needs natural sentences; it can be enabled for classical NLP preprocessing.

## Setup

Requires Python 3.11 (spaCy's compiled dependencies do not build on 3.13+; NumPy is pinned below 2.0 for binary compatibility, see requirements.txt).

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # best name recognition (used automatically if present)
```

The small model `en_core_web_sm` installs via requirements.txt and is used as an automatic fallback on memory-constrained hosts.

## Usage

Command line:

```bash
# Instruction-formatted output (Alpaca style) -> processed_data.json + metrics_report.json
python main.py --input data.csv

# Mirror mode: output matches the input format (CSV in -> processed_data.csv),
# sanitised text only (metadata columns excluded by default for privacy)
python main.py --input data.csv --mode mirror

# Configurability examples
python main.py --input data.json --toxicity-threshold 0.7
python main.py --input data.csv --similarity-threshold 0.85 --instruction "Summarise the following text"
python main.py --input data.csv --lemmatize --remove-stopwords   # opt-in normalisation
```

Web interface:

```bash
streamlit run app.py
```

Upload a CSV or JSON with a `text` column, adjust thresholds in the sidebar, and download the instruction dataset, the format-mirrored sanitised data, and the metrics report.

Evaluation (PII anonymisation recall against token-level BIO ground truth):

```bash
python evaluate_pii.py --original labelled_data.csv --processed processed_data.json
```

## Key design decisions

- **Hybrid PII detection.** Deterministic regexes claim structured entities (emails, international phone numbers, street addresses, @handles) before Presidio's ML recognisers handle context-dependent entities (names, locations). Introduced after a ground-truth recall evaluation of the Presidio-only baseline measured 58.2% strict recall with failures concentrated in street addresses (no recogniser exists), US-centric phone handling, and non-Western name recognition; the hybrid design measures 97.0% on the same benchmark.
- **Ensemble toxicity filtering.** Detoxify screens explicit toxicity across four attributes (toxicity, identity_attack, insult, threat); the ToxiGen RoBERTa specialist (`tomh/toxigen_roberta`) screens implicit hate expressed without toxic vocabulary. A row is removed if either fires, and every removal is attributed by source in the metrics. On a 50-row ToxiGen adversarial benchmark, Detoxify alone removed 2/50 (4%); the ensemble removes 37/50 (74%). The specialist loads lazily and the filter degrades gracefully to Detoxify-only where memory is constrained.
- **Removal, not redaction, for harmful content.** A `[TOXIC CONTENT REMOVED]` placeholder would itself become a training example; flagged rows are excluded instead. PII, by contrast, is redacted in place with `[ENTITY]` placeholders because the surrounding text retains training value.
- **Whole-record privacy in mirror mode.** Only the sanitised text column is exported by default, because source datasets often carry PII in metadata columns (name, email, phone); `--keep-columns` opts out with a warning.
- **Attributed metrics.** Every removed row is attributed to a specific stage (validation, exact duplicate, near duplicate, toxicity by attribute), so nothing disappears from the dataset unaccounted for. The report includes before/after samples for qualitative review.
- **Precision filtering.** DATE_TIME findings are excluded from redaction: durations ("13 years of experience") are not personally identifying and removing them destroys training value. Trade-off (dates of birth) documented in `pii_detector.py`.

## Hosted vs local configuration

The free hosting tier (~2.7 GB RAM) cannot hold the large spaCy model plus two transformers, so the hosted demo runs `en_core_web_sm` and may fall back to Detoxify-only filtering. All reported evaluation figures were produced locally with the full configuration (`en_core_web_lg` + ensemble). Use the "Max rows" sidebar setting for demo-sized runs on the hosted app.

## Known limitations

- Pairwise TF-IDF similarity is O(n^2) in memory: suitable at project scale, not web scale (MinHash/LSH is the scalable alternative).
- English-focused: Presidio, Detoxify, and ToxiGen RoBERTa are English-centric; multilingual data is out of scope.
- Residual detection gaps are characterised in the project report: suffix-less street addresses, bare usernames without @, and character-obfuscated slurs can evade detection. Automated sanitisation should be complemented by human review for high-stakes datasets.
- The `output` field in instruction records is intentionally empty: producing target completions is an annotation task outside a sanitisation pipeline's scope.

## Project structure

```
src/
  main.py             CLI orchestrator
  app.py              Streamlit web interface
  config.py           Central configuration (PipelineConfig)
  ingestion.py        CSV/JSON loading
  validator.py        Schema and row validation
  pii_detector.py     Hybrid regex + Presidio PII anonymisation
  toxicity_filter.py  Detoxify + ToxiGen RoBERTa ensemble
  deduplicator.py     Exact + TF-IDF/cosine near-duplicate removal
  cleaner.py          URL/HTML/whitespace cleaning; opt-in normalisation
  formatter.py        Alpaca-style instruction formatting
  exporter.py         Instruction and format-mirroring export
  metrics.py          Per-stage metrics and qualitative samples
  evaluate_pii.py     Ground-truth recall evaluator (BIO labels)
  requirements.txt
```

## Third-party libraries and attribution

All pipeline logic in this repository is the author's own work. It builds on the following open source components, used unmodified via their public APIs:

- [Microsoft Presidio](https://github.com/microsoft/presidio) (MIT) - PII analysis and anonymisation engines
- [Detoxify](https://github.com/unitaryai/detoxify) (Apache-2.0) - explicit toxicity classification
- [ToxiGen RoBERTa](https://huggingface.co/tomh/toxigen_roberta) - implicit hate classification (Hartvigsen et al., 2022, ACL)
- [spaCy](https://spacy.io) (MIT) - tokenisation, lemmatisation, NER models
- [scikit-learn](https://scikit-learn.org) (BSD-3) - TF-IDF vectorisation and cosine similarity
- [pandas](https://pandas.pydata.org) (BSD-3), [NumPy](https://numpy.org) (BSD-3) - data handling
- [Streamlit](https://streamlit.io) (Apache-2.0) - web interface

Evaluation datasets: a synthetic PII-labelled biography corpus (no real personal data) and a 50-row subset of the ToxiGen benchmark (Hartvigsen et al., 2022), used strictly for filter evaluation and not redistributed.