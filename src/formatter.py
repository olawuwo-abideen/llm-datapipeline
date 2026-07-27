"""Stage 7: Instruction formatting.

Transforms the sanitised dataset into the Alpaca-style
instruction / input / output structure widely used for LLM
supervised fine-tuning.

SCOPE NOTE: the pipeline sanitises inputs; producing 'output' labels
(the target completions) is a labelling task outside the scope of a
sanitisation pipeline, so 'output' is intentionally left empty for a
downstream annotation step. The instruction text is configurable so
the artefact is reusable across tasks.
"""

import pandas as pd


class Formatter:

    def __init__(self, instruction: str = "Process the following text"):
        self.instruction = instruction

    def format(self, df: pd.DataFrame) -> list:
        return [
            {
                "instruction": self.instruction,
                "input": text,
                "output": "",
            }
            for text in df["text"]
        ]
