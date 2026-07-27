"""Stage 6: Text cleaning and linguistic normalisation.

ORDERING NOTE: this stage now runs LAST among the text transformations,
after PII anonymisation and toxicity filtering, because lowercasing and
lemmatisation destroy the signals (capitalisation, sentence structure)
that Presidio and Detoxify depend on.

Lemmatisation and stopword removal are configurable. For many LLM
fine-tuning tasks you actually want to KEEP natural sentences, so
being able to switch normalisation off is part of making the artefact
reusable. Anonymisation placeholders like [EMAIL] are protected from
lemmatisation.
"""

import re

import spacy

nlp = spacy.load("en_core_web_sm")

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")
URL_PATTERN = re.compile(r"http\S+")
HTML_PATTERN = re.compile(r"<.*?>")
WHITESPACE_PATTERN = re.compile(r"\s+")


class Cleaner:

    def __init__(self, lemmatize: bool = True, remove_stopwords: bool = True):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords

    def clean_series(self, texts) -> list:
        """Batched cleaning using spaCy's nlp.pipe, typically 5-10x faster
        than calling clean() row by row, because spaCy processes documents
        in efficient batches instead of one at a time."""
        prepared = []
        all_placeholders = []

        for text in texts:
            if not isinstance(text, str):
                text = ""
            text = URL_PATTERN.sub("", text)
            text = HTML_PATTERN.sub("", text)
            placeholders = PLACEHOLDER_PATTERN.findall(text)
            for i, ph in enumerate(placeholders):
                text = text.replace(ph, f"xxplaceholderxx{i}xx", 1)
            prepared.append(text.lower().strip())
            all_placeholders.append(placeholders)

        if not (self.lemmatize or self.remove_stopwords):
            results = prepared
        else:
            results = []
            for doc in nlp.pipe(prepared, batch_size=256):
                tokens = []
                for token in doc:
                    if token.is_punct:
                        continue
                    if self.remove_stopwords and token.is_stop:
                        continue
                    tokens.append(token.lemma_ if self.lemmatize else token.text)
                results.append(" ".join(tokens))

        cleaned = []
        for text, placeholders in zip(results, all_placeholders):
            for i, ph in enumerate(placeholders):
                text = text.replace(f"xxplaceholderxx{i}xx", ph)
            cleaned.append(WHITESPACE_PATTERN.sub(" ", text).strip())

        return cleaned

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Structural noise removal
        text = URL_PATTERN.sub("", text)
        text = HTML_PATTERN.sub("", text)

        # Protect anonymisation placeholders (e.g. [EMAIL], [PERSON])
        # from being lowercased and lemmatised
        placeholders = PLACEHOLDER_PATTERN.findall(text)
        for i, ph in enumerate(placeholders):
            text = text.replace(ph, f"xxplaceholderxx{i}xx", 1)

        text = text.lower().strip()

        if self.lemmatize or self.remove_stopwords:
            doc = nlp(text)
            tokens = []
            for token in doc:
                if token.is_punct:
                    continue
                if self.remove_stopwords and token.is_stop:
                    continue
                tokens.append(token.lemma_ if self.lemmatize else token.text)
            text = " ".join(tokens)

        # Restore placeholders
        for i, ph in enumerate(placeholders):
            text = text.replace(f"xxplaceholderxx{i}xx", ph)

        return WHITESPACE_PATTERN.sub(" ", text).strip()
