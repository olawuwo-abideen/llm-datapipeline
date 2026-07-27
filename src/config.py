"""Central configuration for the data sanitisation pipeline.

All tunable parameters live here so the pipeline is reusable across
datasets without editing pipeline logic (Objective: flexible, reusable
framework). Values can be overridden from the command line in main.py.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # I/O
    input_path: str = "../datasets/data.csv"
    output_path: str = "processed.json"
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

    # Cleaning
    lemmatize: bool = True
    remove_stopwords: bool = True
