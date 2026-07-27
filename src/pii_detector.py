"""Stage 3: PII detection and anonymisation.

IMPORTANT ORDERING NOTE: this stage must run on the ORIGINAL text,
before lowercasing / lemmatisation. Presidio's NER models rely on
capitalisation and sentence structure to recognise names, locations
and organisations; running it on cleaned text sharply reduces recall.

A regex email fallback is kept as a safety net for addresses Presidio
occasionally misses in noisy text.
"""

import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class PIIDetector:

    EMAIL_PATTERN = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.pii_count = 0
        self.rows_affected = 0

    def anonymize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        found_in_row = 0

        # Regex email fallback
        emails = self.EMAIL_PATTERN.findall(text)
        if emails:
            found_in_row += len(emails)
            text = self.EMAIL_PATTERN.sub("[EMAIL]", text)

        results = self.analyzer.analyze(text=text, language="en")
        found_in_row += len(results)

        if results:
            text = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
            ).text

        if found_in_row:
            self.pii_count += found_in_row
            self.rows_affected += 1

        return text
