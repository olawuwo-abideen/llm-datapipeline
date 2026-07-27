"""Stage 8: Export.

Writes the formatted dataset to disk. The output path is a parameter
(the earlier version hardcoded 'processed.json' and was never called
from main.py, so the pipeline produced no artefact).
"""

import json
from pathlib import Path


class Exporter:

    def save(self, data: list, output_path: str = "processed.json") -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return str(path)
