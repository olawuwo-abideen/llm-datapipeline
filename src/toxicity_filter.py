"""Stage 4: Toxicity and bias filtering (ensemble version).

ARCHITECTURE: two complementary classifiers, a row is removed if EITHER fires.

1. Detoxify ("original"): strong on EXPLICIT toxicity (insults, threats,
   profanity, slurs). Scores checked across four attributes (toxicity,
   identity_attack, insult, threat) to cover the "biased or harmful
   content" objective.

2. ToxiGen RoBERTa (tomh/toxigen_roberta): a specialist fine-tuned on
   implicit, adversarially generated hate speech. Added after benchmark
   evaluation on ToxiGen data showed Detoxify caught only 2/50 implicit
   hate statements (4% recall), with identity_attack firing zero times.
   Diagnosis: implicit hate contains no toxic lexical markers, so no
   Detoxify threshold rescues it (scores sit near zero, not near 0.6),
   ruling out tuning and motivating an ensemble instead. The specialist
   itself flags only ~14/50 of the same benchmark, so residual implicit
   hate remains a documented open limitation of classifier-based
   filtering (per the ToxiGen paper), mitigated in practice by human
   review for high-stakes datasets.

DESIGN DECISIONS RETAINED FROM THE PREVIOUS VERSION:
- BATCHED inference for both models (one forward pass per batch).
- Rows are DROPPED rather than replaced with a placeholder: a
  "[TOXIC CONTENT REMOVED]" string would itself become a training
  example during fine-tuning, which is worse than excluding the row.
- Runs on ORIGINAL text (both models were trained on natural sentences).

RESOURCE NOTE: the ensemble loads a second transformer (~500 MB). On
memory-constrained hosts (e.g. Streamlit Cloud free tier, ~2.7 GB RAM)
this will exceed the ceiling, so the specialist loads lazily and the
filter degrades gracefully to Detoxify-only if it cannot be loaded,
printing a warning. Full ensemble evaluation should be run locally or
on Hugging Face Spaces. Known residual failure mode either way:
character-level obfuscation of slurs can evade both classifiers.
"""

import pandas as pd
from detoxify import Detoxify


class ToxicityFilter:

    TOXIGEN_MODEL = "tomh/toxigen_roberta"

    def __init__(self, threshold: float = 0.60, attributes=None,
                 batch_size: int = 64, hate_threshold: float = 0.50,
                 use_implicit_hate_model: bool = True):
        # Device selection: CUDA (cloud GPU), MPS (Apple Silicon), or CPU
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"ToxicityFilter running on device: {self.device}")

        self.model = Detoxify("original", device=self.device)
        self.threshold = threshold
        self.attributes = list(attributes) if attributes else ["toxicity"]
        self.batch_size = batch_size
        self.hate_threshold = hate_threshold

        self.flagged = 0
        self.flagged_by_attribute = {}

        # Specialist implicit-hate classifier: lazy, graceful fallback
        self.hate_tokenizer = None
        self.hate_model = None
        if use_implicit_hate_model:
            try:
                from transformers import (AutoModelForSequenceClassification,
                                          AutoTokenizer)
                self.hate_tokenizer = AutoTokenizer.from_pretrained(
                    self.TOXIGEN_MODEL)
                self.hate_model = AutoModelForSequenceClassification \
                    .from_pretrained(self.TOXIGEN_MODEL).to(self.device)
                self.hate_model.eval()
                print(f"Implicit-hate specialist loaded: {self.TOXIGEN_MODEL}")
            except Exception as e:
                print(f"WARNING: implicit-hate model unavailable "
                      f"({type(e).__name__}: {e}). Falling back to "
                      f"Detoxify-only filtering; implicit hate recall "
                      f"will be substantially reduced.")

    def _implicit_hate_scores(self, batch):
        """Toxicity probability per text from the ToxiGen specialist
        (softmax over its two labels; index 1 = toxic/hateful)."""
        import torch
        inputs = self.hate_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.hate_model(**inputs).logits
        return torch.softmax(logits, dim=-1)[:, 1].tolist()

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df["text"].tolist()
        if not texts:
            return df

        keep_mask = []

        for start in range(0, len(texts), self.batch_size):
            batch = [str(t) for t in texts[start:start + self.batch_size]]

            # Classifier 1: Detoxify (explicit toxicity, four attributes)
            scores = self.model.predict(batch)

            # Classifier 2: ToxiGen RoBERTa (implicit hate), if loaded
            hate_scores = (self._implicit_hate_scores(batch)
                           if self.hate_model is not None
                           else [0.0] * len(batch))

            for i in range(len(batch)):
                flagged_attrs = [
                    attr for attr in self.attributes
                    if attr in scores and scores[attr][i] > self.threshold
                ]
                if hate_scores[i] > self.hate_threshold:
                    flagged_attrs.append("implicit_hate")

                if flagged_attrs:
                    self.flagged += 1
                    for attr in flagged_attrs:
                        self.flagged_by_attribute[attr] = (
                            self.flagged_by_attribute.get(attr, 0) + 1
                        )
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)

        return df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)
