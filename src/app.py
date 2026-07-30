"""Streamlit web interface for the data sanitisation pipeline.

Run locally:
    streamlit run app.py

STATE HANDLING NOTE: Streamlit reruns the whole script on every widget
interaction, including download-button clicks. Pipeline results are
therefore stored in st.session_state after a run and rendered from
session state on every rerun, so downloading one file does not wipe
the results and force a re-run of the pipeline.

DOWNLOADS: three outputs are offered after a run:
1. Instruction dataset (Alpaca-style JSON) for LLM fine-tuning
2. Sanitised dataset mirroring the INPUT format (CSV in -> CSV out,
   JSON in -> JSON out), containing only the sanitised text column;
   metadata columns are excluded because they may carry unsanitised PII
3. Metrics report (JSON)
"""

import io
import json
import time
from pathlib import Path

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
    lemmatize = st.checkbox(
        "Lemmatise", value=False,
        help="Off by default: fine-tuning data needs natural sentences. "
             "Enable for classical NLP preprocessing.")
    remove_stopwords = st.checkbox(
        "Remove stopwords", value=False,
        help="Off by default; enable for classical NLP preprocessing.")
    max_rows = st.number_input(
        "Max rows to process (0 = all)", min_value=0, value=0, step=100,
        help="Useful for quick demos on free CPU hosting",
    )

uploaded = st.file_uploader("Upload dataset", type=["csv", "json"])

if uploaded and st.button("Run pipeline", type="primary"):
    # Ingest
    input_extension = Path(uploaded.name).suffix.lower()
    if input_extension == ".csv":
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
    elapsed = time.time() - start
    progress.progress(100, text=f"Done in {elapsed:.1f}s")

    reduction = ((len(before) - len(after)) / len(before) * 100) if len(before) else 0

    # Sanitised data mirroring the input format: text column only, since
    # metadata columns may carry unsanitised PII (name, email, phone, ...)
    text_only = after[["text"]]
    if input_extension == ".csv":
        buf = io.StringIO()
        text_only.to_csv(buf, index=False)
        mirror_data = buf.getvalue()
        mirror_filename = "processed_data.csv"
        mirror_mime = "text/csv"
    else:
        mirror_data = text_only.to_json(orient="records", indent=4,
                                        force_ascii=False)
        mirror_filename = "processed_data.json"
        mirror_mime = "application/json"

    # PERSIST results in session state so download clicks (which rerun
    # the script) do not wipe them
    st.session_state["results"] = {
        "formatted_json": json.dumps(formatted, indent=4, ensure_ascii=False),
        "mirror_data": mirror_data,
        "mirror_filename": mirror_filename,
        "mirror_mime": mirror_mime,
        "metrics": {
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
        },
        "samples": {
            "before": [str(before["text"].iloc[i])[:200]
                       for i in range(min(5, len(after), len(before)))],
            "after": [str(after["text"].iloc[i])[:200]
                      for i in range(min(5, len(after), len(before)))],
        },
        "elapsed": round(elapsed, 1),
        "source_file": uploaded.name,
    }

# ----- Results: rendered from session state on EVERY rerun -----
if "results" in st.session_state:
    r = st.session_state["results"]
    m = r["metrics"]

    st.success(f"Pipeline results for '{r['source_file']}' "
               f"(completed in {r['elapsed']}s). "
               "Results persist until you run the pipeline again.")

    st.subheader("Pipeline metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original rows", m["original_rows"])
    c2.metric("Final rows", m["final_rows"])
    c3.metric("PII entities anonymised", m["pii_entities_anonymised"])
    c4.metric("Toxic/biased rows removed", m["rows_removed_as_toxic_or_biased"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Dropped by validation", m["dropped_by_validation"])
    c6.metric("Exact duplicates", m["exact_duplicates_removed"])
    c7.metric("Near duplicates", m["near_duplicates_removed"])
    c8.metric("Reduction %", f"{m['reduction_percent']}%")

    if m["toxicity_breakdown_by_attribute"]:
        st.write("Toxicity breakdown by attribute:",
                 m["toxicity_breakdown_by_attribute"])

    st.subheader("Before / after samples")
    st.dataframe(pd.DataFrame(r["samples"]), use_container_width=True)

    st.subheader("Download outputs")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Instruction dataset (JSON)",
        r["formatted_json"],
        file_name="processed_data.json",
        mime="application/json",
        key="dl_dataset",
        help="Alpaca-style instruction/input/output records for fine-tuning",
    )
    d2.download_button(
        f"Sanitised data ({r['mirror_filename'].split('.')[-1].upper()})",
        r["mirror_data"],
        file_name=r["mirror_filename"],
        mime=r["mirror_mime"],
        key="dl_mirror",
        help="Sanitised text in the same format as your input file",
    )
    d3.download_button(
        "Metrics report (JSON)",
        json.dumps(m, indent=4, ensure_ascii=False),
        file_name="metrics_report.json",
        mime="application/json",
        key="dl_metrics",
    )
