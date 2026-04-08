import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import ijson

from .base import BaseExtractor


class JSONExtractor(BaseExtractor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.encoding = self.config.get("encoding", "utf-8")
        self.normalize = self.config.get("normalize", True)
        self.max_level = self.config.get("max_level", 3)
        self.stream = self.config.get("stream", False)
        self.array_path = self.config.get("array_path")

    def _extract_impl(self, file_path: Union[str, Path], **kwargs):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON não encontrado: {file_path}")

        self.logger.info(f"lendo JSON: {file_path.name}")

        if file_path.suffix.lower() == ".jsonl" or self.config.get("json_lines"):
            return self._read_jsonl(file_path)

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if self.stream or size_mb > 100:
            self.logger.info(f"streaming arquivo grande ({size_mb:.1f} MB)")
            return self._read_streaming(file_path)

        try:
            with file_path.open(encoding=self.encoding) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {e}")

        if self.normalize and isinstance(data, (dict, list)):
            return self._normalize(data)
        return data

    def _read_jsonl(self, file_path: Path) -> pd.DataFrame:
        rows = []
        with file_path.open(encoding=self.encoding) as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self.logger.warning(f"linha {n} ignorada: {e}")
        self.logger.info(f"JSONL: {len(rows)} linhas válidas")
        if self.normalize:
            return pd.json_normalize(rows, max_level=self.max_level)
        return pd.DataFrame(rows)

    def _read_streaming(self, file_path: Path) -> List[Dict]:
        data = []
        with file_path.open("rb") as f:
            for obj in ijson.items(f, self.array_path or "item"):
                data.append(obj)
        self.logger.info(f"streaming: {len(data)} itens")
        return data

    def _normalize(self, data: Union[Dict, List]) -> pd.DataFrame:
        if isinstance(data, list):
            return pd.json_normalize(data, max_level=self.max_level)
        for key, value in data.items():
            if isinstance(value, list):
                self.logger.info(f"normalizando array em '{key}'")
                return pd.json_normalize(value, max_level=self.max_level)
        return pd.json_normalize([data], max_level=self.max_level)
