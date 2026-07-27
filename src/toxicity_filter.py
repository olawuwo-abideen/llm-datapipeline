"""Stage 4: Toxicity and bias filtering.

Two improvements over the earlier version:

1. BATCHED inference. Detoxify accepts a list of texts, and batching
   the whole column through the model is dramatically faster than a
   per-row .apply() (one forward pass per batch instead of per row).

2. MULTI-ATTRIBUTE filtering. The project objectives require removing
   "biased, toxic, or harmful" content. Filtering only on the
   'toxicity' score misses identity-based attacks; Detoxify already
   returns identity_attack, insult and threat scores in the same call,
   so these are checked too at no extra cost.

Like PII detection, this runs on the ORIGINAL text: the model was
trained on natural sentences, so scores on lemmatised, stopword-free
text are unreliable.

Rows are dropped rather than replaced with a placeholder string. A
"[TOXIC CONTENT REMOVED]" placeholder would itself become a training
example during fine-tuning, which is worse than simply excluding the
row.
"""

import pandas as pd
from detoxify import Detoxify


class ToxicityFilter:

    def __init__(self, threshold: float = 0.60, attributes=None, batch_size: int = 64):
        # Use GPU when available (CUDA on Colab/cloud, MPS on Apple Silicon),
        # which speeds up the transformer inference dramatically.
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"ToxicityFilter running on device: {device}")
        self.model = Detoxify("original", device=device)
        self.threshold = threshold
        self.attributes = list(attributes) if attributes else ["toxicity"]
        self.batch_size = batch_size
        self.flagged = 0
        self.flagged_by_attribute = {}

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df["text"].tolist()

        if not texts:
            return df

        keep_mask = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            scores = self.model.predict(batch)

            for i in range(len(batch)):
                flagged_attrs = [
                    attr for attr in self.attributes
                    if attr in scores and scores[attr][i] > self.threshold
                ]

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
