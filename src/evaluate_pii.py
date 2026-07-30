"""PII anonymisation recall evaluator.

Scores the sanitisation pipeline's output against a labelled dataset
containing token-level BIO ground-truth annotations (e.g. B-NAME_STUDENT,
I-NAME_STUDENT, B-PHONE_NUM, B-STREET_ADDRESS, ...), producing per-entity-type
recall figures for the dissertation's evaluation chapter.

Usage:
    python evaluate_pii.py --original data.csv --processed processed.json
    python evaluate_pii.py --original data.csv --processed processed.json \
        --report pii_recall_report.json

Method:
1. Ground-truth entities are reconstructed from the dataset's `tokens` and
   `labels` columns (BIO scheme). Using the BIO labels rather than the
   metadata columns (name/email/phone/...) is deliberate: BIO labels mark
   only PII that actually appears in the text, so the evaluation is not
   distorted by metadata values the generator never inserted.
2. Each entity is broken into alphanumeric chunks (case-insensitive), and
   the corresponding processed output row is searched for each chunk as a
   whole word. This makes the check robust to the pipeline's lowercasing
   and lemmatisation.
3. Each unique entity per row is classified by the fraction of its chunks
   still present in the output:
       removed  -> 0% of chunks found   (fully anonymised)
       partial  -> >0% and <50% found   (fragment leaked, e.g. area code)
       leaked   -> >=50% found          (substantially survived)
4. Strict recall per entity type = removed / total. The report also gives
   a token-level leak rate and lists every non-removed entity with the
   exact chunks that survived, for qualitative discussion.

Assumption: the processed JSON has the same number of rows, in the same
order, as the original dataset (true when validation/dedup/toxicity drop
nothing). The script refuses to run if the counts differ, rather than
misalign rows silently.

Scoring caveat (deliberate): when two entities share chunks (e.g. the
email arina-sun@gmail.net contains the name Arina Sun), a leak of one
can cause the other to score as partially leaked even if its own
placeholder was correctly inserted. The evaluation is therefore
CONSERVATIVE: it can overstate leakage but never understate it, which
is the safe direction of error for a privacy evaluation. Report this
as a methodological note in the evaluation chapter.
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict

import pandas as pd

CHUNK_PATTERN = re.compile(r"[a-z0-9]+")
PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")


def output_chunks(text):
    """Chunks of a processed output row, with anonymisation placeholders
    removed first. Without this, [STREET_ADDRESS] would contribute the
    chunks 'street' and 'address', falsely matching entities that contain
    those generic words and overstating leakage."""
    return CHUNK_PATTERN.findall(PLACEHOLDER_PATTERN.sub(" ", text).lower())


def extract_entities(tokens, labels):
    """Reconstruct (entity_type, entity_text) spans from BIO labels."""
    entities = []
    current_type, current_tokens = None, []

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_type:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(token)
        else:
            if current_type:
                entities.append((current_type, " ".join(current_tokens)))
            current_type, current_tokens = None, []

    if current_type:
        entities.append((current_type, " ".join(current_tokens)))

    return entities


def chunks_of(text):
    """Lowercase alphanumeric chunks of an entity or document."""
    return CHUNK_PATTERN.findall(text.lower())


def classify(entity_text, output_word_set):
    """Return (status, leaked_chunks, fraction) for one entity."""
    chunks = chunks_of(entity_text)
    if not chunks:
        return "removed", [], 0.0

    leaked = [c for c in chunks if c in output_word_set]
    fraction = len(leaked) / len(chunks)

    if fraction == 0:
        return "removed", leaked, fraction
    if fraction < 0.5:
        return "partial", leaked, fraction
    return "leaked", leaked, fraction


def evaluate(original_csv, processed_json, report_path):
    df = pd.read_csv(original_csv)

    for col in ("tokens", "labels"):
        if col not in df.columns:
            sys.exit(f"ERROR: original dataset has no '{col}' column; "
                     "BIO ground-truth labels are required.")

    with open(processed_json, encoding="utf-8") as f:
        processed = json.load(f)

    if len(processed) != len(df):
        sys.exit(
            f"ERROR: row count mismatch (original {len(df)}, "
            f"processed {len(processed)}). The evaluator matches rows by "
            "position, so run the pipeline on this dataset with no rows "
            "dropped, or align the files first."
        )

    stats = defaultdict(lambda: {"total": 0, "removed": 0, "partial": 0,
                                 "leaked": 0, "fractions": []})
    leak_details = []

    for i, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"])
        labels = ast.literal_eval(row["labels"])
        entities = extract_entities(tokens, labels)

        output_text = processed[i].get("input", "")
        output_words = set(output_chunks(output_text))

        # Deduplicate repeated mentions of the same entity within a row:
        # the output is a single string, so every mention scores identically
        # and counting each would inflate the totals.
        seen = set()
        for etype, etext in entities:
            key = (etype, " ".join(chunks_of(etext)))
            if key in seen:
                continue
            seen.add(key)

            status, leaked_chunks, fraction = classify(etext, output_words)

            s = stats[etype]
            s["total"] += 1
            s[status] += 1
            s["fractions"].append(fraction)

            if status != "removed":
                leak_details.append({
                    "row": int(i),
                    "entity_type": etype,
                    "ground_truth": etext,
                    "status": status,
                    "surviving_chunks": leaked_chunks,
                    "chunk_leak_fraction": round(fraction, 2),
                })

    # Build the report
    per_type = {}
    for etype in sorted(stats):
        s = stats[etype]
        per_type[etype] = {
            "total_entities": s["total"],
            "fully_removed": s["removed"],
            "partial_leaks": s["partial"],
            "leaked": s["leaked"],
            "strict_recall_percent": round(100 * s["removed"] / s["total"], 1),
            "mean_token_leak_percent": round(
                100 * sum(s["fractions"]) / len(s["fractions"]), 1),
        }

    total = sum(s["total"] for s in stats.values())
    removed = sum(s["removed"] for s in stats.values())
    report = {
        "original_dataset": original_csv,
        "processed_dataset": processed_json,
        "rows_evaluated": len(df),
        "overall": {
            "total_unique_entities": total,
            "fully_removed": removed,
            "strict_recall_percent": round(100 * removed / total, 1) if total else 0,
        },
        "per_entity_type": per_type,
        "leak_details": leak_details,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # Console summary
    print("\n===== PII ANONYMISATION RECALL =====")
    print(f"{'Entity type':<20}{'Total':>6}{'Removed':>9}"
          f"{'Partial':>9}{'Leaked':>8}{'Recall %':>10}")
    for etype, s in per_type.items():
        print(f"{etype:<20}{s['total_entities']:>6}{s['fully_removed']:>9}"
              f"{s['partial_leaks']:>9}{s['leaked']:>8}"
              f"{s['strict_recall_percent']:>10}")
    print(f"\nOverall strict recall: "
          f"{report['overall']['strict_recall_percent']}% "
          f"({removed}/{total} entities fully removed)")
    print(f"Leak details ({len(leak_details)} entities) saved to: "
          f"{report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score PII anonymisation recall against BIO ground truth")
    parser.add_argument("--original", required=True,
                        help="Original labelled CSV (with tokens and labels columns)")
    parser.add_argument("--processed", required=True,
                        help="Pipeline output JSON (processed.json)")
    parser.add_argument("--report", default="pii_recall_report.json",
                        help="Where to write the JSON report")
    args = parser.parse_args()

    evaluate(args.original, args.processed, args.report)
