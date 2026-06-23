from ingestion import DataIngestion
from validator import Validator
from cleaner import Cleaner
from deduplicator import Deduplicator
from pii_detector import PIIDetector
from toxicity_filter import ToxicityFilter
from metrics import Metrics


def run():

    ingestion = DataIngestion()
    validator = Validator()
    cleaner = Cleaner()
    dedup = Deduplicator()
    pii = PIIDetector()
    toxic = ToxicityFilter()
    metrics = Metrics()

    df = ingestion.load("../datasets/raw_dataset.csv")

    # save original data
    before = df.copy()

    validator.validate(df)

    df["text"] = df["text"].apply(cleaner.clean)

    df = dedup.remove_duplicates(df)

    df["text"] = df["text"].apply(pii.anonymize)

    df["text"] = df["text"].apply(toxic.filter_text)

    # save processed data
    after = df.copy()

    # run metrics
    metrics.generate(
        before,
        after,
        pii_found=2,
        toxic_flagged=1
    )

    print(df)


run()