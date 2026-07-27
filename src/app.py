"""Streamlit web interface for the data sanitisation pipeline.

Run locally:
    streamlit run app.py

Or deploy free on Hugging Face Spaces (Streamlit SDK): upload this
folder to a new Space and it serves automatically at a public URL.

The UI wraps the same modules used by the CLI (main.py), so the
dissertation artefact has one pipeline with two interfaces.
"""

import json
import time

import pandas as pd
import streamlit as st

from cleaner import Cleaner
from deduplicator import Deduplicator
from formatter import Formatter
from pii_detector import PIIDetector
from toxicity_filter import ToxicityFilter
from validator import Validator

st.set_page_config(page_title="LLM Data Sanitisation Pipeline", layout="wide")

st.title("Data Sanitisation Pipeline for LLM Fine-Tuning")
st.caption(
    "Upload a CSV or JSON dataset with a 'text' column. The pipeline validates, "
    "anonymises PII, filters toxic and biased content, removes duplicates, "
    "normalises text, and exports an instruction-formatted dataset."
)


# Heavy models are loaded once and cached across reruns
@st.cache_resource
def load_pii_engines():
    detector = PIIDetector()
    return detector.analyzer, detector.anonymizer


@st.cache_resource
def load_toxicity_model(threshold, attributes):
    return ToxicityFilter(threshold=threshold, attributes=attributes)


with st.sidebar:
    st.header("Settings")
    similarity_threshold = st.slider(
        "Near-duplicate similarity threshold", 0.50, 1.00, 0.90, 0.01
    )
    toxicity_threshold = st.slider(
        "Toxicity / bias threshold", 0.10, 1.00, 0.60, 0.05
    )
    attributes = st.multiselect(
        "Filtered attributes",
        ["toxicity", "identity_attack", "insult", "threat"],
        default=["toxicity", "identity_attack", "insult", "threat"],
    )
    instruction_text = st.text_input(
        "Instruction for formatted records", "Process the following text"
    )
    lemmatize = st.checkbox("Lemmatise", value=True)
    remove_stopwords = st.checkbox("Remove stopwords", value=True)
    max_rows = st.number_input(
        "Max rows to process (0 = all)", min_value=0, value=0, step=100,
        help="Useful for quick demos on free CPU hosting",
    )

uploaded = st.file_uploader("Upload dataset", type=["csv", "json"])

if uploaded and st.button("Run pipeline", type="primary"):
    # Ingest
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_json(uploaded)

    if max_rows:
        df = df.head(int(max_rows))

    before = df.copy()
    progress = st.progress(0, text="Validating...")
    start = time.time()

    # Validate
    validator = Validator()
    try:
        df = validator.validate(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    progress.progress(15, text="Anonymising PII...")

    # PII (cached engines, wired into a fresh detector for correct counters)
    pii = PIIDetector.__new__(PIIDetector)
    pii.analyzer, pii.anonymizer = load_pii_engines()
    pii.pii_count = 0
    pii.rows_affected = 0
    df["text"] = df["text"].apply(pii.anonymize)
    progress.progress(45, text="Filtering toxic and biased content...")

    # Toxicity / bias (cached model)
    toxic = load_toxicity_model(toxicity_threshold, tuple(attributes))
    toxic.threshold = toxicity_threshold
    toxic.attributes = list(attributes)
    toxic.flagged = 0
    toxic.flagged_by_attribute = {}
    df = toxic.filter_dataframe(df)
    progress.progress(70, text="Removing duplicates...")

    # Dedup
    dedup = Deduplicator(threshold=similarity_threshold)
    df = dedup.remove_duplicates(df)
    progress.progress(85, text="Cleaning and normalising...")

    # Clean (batched)
    cleaner = Cleaner(lemmatize=lemmatize, remove_stopwords=remove_stopwords)
    df["text"] = cleaner.clean_series(df["text"].tolist())
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    after = df.copy()

    # Format
    formatted = Formatter(instruction=instruction_text).format(df)
    progress.progress(100, text=f"Done in {time.time() - start:.1f}s")

    # ----- Results -----
    st.subheader("Pipeline metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original rows", len(before))
    c2.metric("Final rows", len(after))
    c3.metric("PII entities anonymised", pii.pii_count)
    c4.metric("Toxic/biased rows removed", toxic.flagged)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Dropped by validation", validator.rows_dropped)
    c6.metric("Exact duplicates", dedup.exact_removed)
    c7.metric("Near duplicates", dedup.near_removed)
    reduction = ((len(before) - len(after)) / len(before) * 100) if len(before) else 0
    c8.metric("Reduction %", f"{reduction:.1f}%")

    if toxic.flagged_by_attribute:
        st.write("Toxicity breakdown by attribute:", toxic.flagged_by_attribute)

    st.subheader("Before / after samples")
    n = min(5, len(after), len(before))
    st.dataframe(pd.DataFrame({
        "before": [str(before["text"].iloc[i])[:200] for i in range(n)],
        "after": [str(after["text"].iloc[i])[:200] for i in range(n)],
    }), use_container_width=True)

    st.subheader("Download outputs")
    metrics_report = {
        "original_rows": len(before),
        "final_rows": len(after),
        "reduction_percent": round(reduction, 2),
        "dropped_by_validation": validator.rows_dropped,
        "exact_duplicates_removed": dedup.exact_removed,
        "near_duplicates_removed": dedup.near_removed,
        "pii_entities_anonymised": pii.pii_count,
        "rows_containing_pii": pii.rows_affected,
        "rows_removed_as_toxic_or_biased": toxic.flagged,
        "toxicity_breakdown_by_attribute": toxic.flagged_by_attribute,
    }
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download formatted dataset (JSON)",
        json.dumps(formatted, indent=4, ensure_ascii=False),
        file_name="processed.json",
        mime="application/json",
    )
    d2.download_button(
        "Download metrics report (JSON)",
        json.dumps(metrics_report, indent=4, ensure_ascii=False),
        file_name="metrics_report.json",
        mime="application/json",
    )
