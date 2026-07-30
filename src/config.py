"""Central configuration for the data sanitisation pipeline.

All tunable parameters live here so the pipeline is reusable across
datasets without editing pipeline logic. Values can be overridden from
the command line in main.py.

DESIGN REFINEMENT (evaluation-driven): linguistic normalisation
(lemmatisation and stopword removal) now defaults to OFF. The
pipeline's primary purpose is preparing data for LLM fine-tuning,
where natural sentences are required; evaluation showed that
normalising by default produced unreadable output for that primary
use case. Normalisation remains available as an explicit opt-in
(--lemmatize / --remove-stopwords) for classical NLP workflows such
as TF-IDF, topic modelling, or bag-of-words classification.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # I/O
    input_path: str = "../datasets/data.csv"
    output_path: str = "processed_data.json"
    metrics_path: str = "metrics_report.json"

    # Deduplication
    similarity_threshold: float = 0.90

    # Toxicity / bias filtering.
    # Detoxify returns several scores per text. Filtering on
    # identity_attack, insult and threat (not just toxicity) covers the
    # "biased or harmful content" part of the project objectives.
    toxicity_threshold: float = 0.60
    filtered_attributes: tuple = (
        "toxicity",
        "identity_attack",
        "insult",
        "threat",
    )

    # Formatting
    instruction_text: str = "Process the following text"

    # Cleaning: normalisation is OPT-IN (defaults off; see module docstring)
    lemmatize: bool = False
    remove_stopwords: bool = False
