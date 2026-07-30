"""Stage 3: PII detection and anonymisation (improved hybrid version).

ARCHITECTURE: deterministic regex detection layered over ML-based NER.
Regex fallbacks run first and claim high-confidence patterns (emails,
phone numbers, street addresses, usernames); Presidio's NER then
handles context-dependent entities (names, locations, organisations).

This hybrid design responds directly to the recall evaluation of the
baseline pipeline (58.2% strict recall), whose failures concentrated in:
  1. STREET_ADDRESS 15.4% recall: Presidio has no street recogniser
     -> dedicated street-address regex added
  2. PHONE_NUM 41.7% recall: US-centric recognisers mislabelled
     international numbers as [US_SSN]/[US_ITIN]/[DATE_TIME], leaking
     area codes -> generalised international phone regex added
  3. NAME_STUDENT 76.2% recall: the small spaCy model missed
     non-Western names (e.g. Arina Sun, Baha)
     -> upgraded to en_core_web_lg where installed, with graceful
        fallback to en_core_web_sm
  4. USERNAME 33.3% recall: no handle detection
     -> @handle regex added (bare usernames without @ remain a
        documented residual limitation: they are context-dependent)

PRECISION REFINEMENT (evaluation-driven, post-recall):
After recall reached 97.0%, qualitative review showed over-redaction:
temporal expressions such as "13 years of experience", "last year" and
"several months" were being redacted as [DATE_TIME]. Durations are not
personally identifying, and removing them destroys the training value
of the text, so DATE_TIME results are now filtered out of Presidio's
findings (see EXCLUDED_ENTITY_TYPES).
  Trade-off accepted and documented: specific dates of birth would no
  longer be caught by this route; a DOB-specific regex is the targeted
  mitigation and is noted as future work.
  ORGANIZATION is deliberately KEPT despite occasional mislabelling
  (e.g. equipment names redacted as [ORGANIZATION]) because it is the
  route by which hard-to-recognise personal names (Arina Sun, Baha)
  were caught: label accuracy is sacrificed for anonymisation coverage.

ORDERING NOTE: this stage must run on the ORIGINAL text, before
lowercasing / lemmatisation, because both the NER models and these
regexes rely on natural formatting.

PRECISION TRADE-OFFS (documented deliberately):
- The street regex requires a street-type suffix, so suffix-less
  addresses (e.g. "6420 Via Baron") can still slip through to NER.
- The phone regex requires a separator or a leading +/0/( so that bare
  long numbers ("12345678 units") are not over-redacted; contiguous
  numbers in phone convention ("07400123456") are still caught.
- In sanitisation, marginal over-redaction is preferred to leakage,
  except where the redacted category is not identifying at all
  (the DATE_TIME case above).

Install the large model for best name recall:
    python -m spacy download en_core_web_lg
On memory-constrained hosts (e.g. Streamlit Cloud free tier) the small
model is used automatically if the large one is absent.
"""

import re

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine


def _build_analyzer(phone_regions):
    """Prefer en_core_web_lg for stronger name NER; fall back to sm."""
    import spacy.util

    model = "en_core_web_lg" if spacy.util.is_package("en_core_web_lg") \
        else "en_core_web_sm"
    print(f"PIIDetector using spaCy model: {model}")

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    })
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    registry.remove_recognizer("PhoneRecognizer")
    registry.add_recognizer(PhoneRecognizer(supported_regions=phone_regions))

    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


class PIIDetector:

    EMAIL_PATTERN = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    # Number + up to three words + street-type suffix + optional direction
    STREET_PATTERN = re.compile(
        r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s+){0,3}"
        r"(?:street|st|avenue|ave|road|rd|drive|dr|way|circle|terrace|"
        r"lane|ln|court|ct|boulevard|blvd|place|track|ferry|views?)"
        r"(?:\s+(?:north|south|east|west|northeast|northwest|"
        r"southeast|southwest))?\b",
        re.IGNORECASE,
    )

    # Digit groups with separators; validated in code for digit count
    # (8-15, the E.164 range) and phone-like lead (+, 0, () or separator
    PHONE_CANDIDATE_PATTERN = re.compile(
        r"\+?\(?\d{1,4}\)?(?:[\s.\-]?\d{2,7}){1,4}"
    )

    # Social handles: @name, letters/digits/underscores/dots
    USERNAME_PATTERN = re.compile(r"@\w[\w.]*\w")

    # Presidio inserts placeholders in angle brackets, e.g. <PERSON>
    PRESIDIO_PLACEHOLDER_PATTERN = re.compile(r"<([A-Z_]+)>")

    # Regions for Presidio's phone recognition; adjust to your data
    PHONE_REGIONS = ["GB", "US", "NG", "IN", "ZA", "CN", "BR"]

    # Entity types detected by Presidio but NOT redacted, because they
    # are not personally identifying and redacting them destroys the
    # text's training value (e.g. durations like "13 years of
    # experience" tagged as DATE_TIME). See module docstring for the
    # accepted trade-off regarding dates of birth.
    EXCLUDED_ENTITY_TYPES = ("DATE_TIME",)

    def __init__(self):
        self.analyzer = _build_analyzer(self.PHONE_REGIONS)
        self.anonymizer = AnonymizerEngine()
        self.pii_count = 0
        self.rows_affected = 0

    def _replace_phones(self, text):
        """Replace phone-like numbers, filtering false positives."""
        count = 0
        result, last_end = [], 0
        for m in self.PHONE_CANDIDATE_PATTERN.finditer(text):
            candidate = m.group()
            digits = sum(c.isdigit() for c in candidate)
            phone_like = (
                8 <= digits <= 15
                and (candidate[0] in "+0("
                     or any(c in " .-()" for c in candidate))
            )
            if phone_like:
                result.append(text[last_end:m.start()])
                result.append("[PHONE_NUMBER]")
                last_end = m.end()
                count += 1
        result.append(text[last_end:])
        return "".join(result), count

    def anonymize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        found_in_row = 0

        # 1. Emails (before usernames, since emails contain @)
        emails = self.EMAIL_PATTERN.findall(text)
        if emails:
            found_in_row += len(emails)
            text = self.EMAIL_PATTERN.sub("[EMAIL]", text)

        # 2. Social handles
        handles = self.USERNAME_PATTERN.findall(text)
        if handles:
            found_in_row += len(handles)
            text = self.USERNAME_PATTERN.sub("[USERNAME]", text)

        # 3. Phone numbers (international, with false-positive filtering)
        text, phone_count = self._replace_phones(text)
        found_in_row += phone_count

        # 4. Street addresses
        streets = self.STREET_PATTERN.findall(text)
        if streets:
            found_in_row += len(streets)
            text = self.STREET_PATTERN.sub("[STREET_ADDRESS]", text)

# 5. ML-based NER for context-dependent entities (names, etc.)
        # score_threshold filters low-confidence detections to reduce
        # false-positive redactions (precision), at some recall risk
        results = self.analyzer.analyze(
            text=text, language="en", score_threshold=0.4
        )

        # PRECISION FILTER: drop entity types that are not personally
        # identifying (durations/dates tagged as DATE_TIME), so that
        # sentences like "13 years of experience" survive intact.
        results = [
            r for r in results
            if r.entity_type not in self.EXCLUDED_ENTITY_TYPES
        ]

        found_in_row += len(results)

        if results:
            text = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
            ).text

            # Presidio uses <ENTITY> placeholders, but the cleaner's
            # HTML-stripping regex (<.*?>) would delete them downstream.
            # Convert to [ENTITY] so they are protected like [EMAIL].
            text = self.PRESIDIO_PLACEHOLDER_PATTERN.sub(r"[\1]", text)

        if found_in_row:
            self.pii_count += found_in_row
            self.rows_affected += 1

        return text
