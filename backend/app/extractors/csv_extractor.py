from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .base import BaseExtractor


class CSVExtractor(BaseExtractor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.delimiter = self.config.get("delimiter", ",")
        self.encoding = self.config.get("encoding", "utf-8")
        self.header = self.config.get("header", 0)
        self.parse_dates = self.config.get("parse_dates")
        self.dtype = self.config.get("dtype")
        self.chunksize = self.config.get("chunksize")
        self.na_values = self.config.get("na_values")

    def _read_params(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        params = {
            "filepath_or_buffer": file_path,
            "sep": self.delimiter,
            "encoding": self.encoding,
            "header": self.header,
            "dtype": self.dtype,
            "parse_dates": self.parse_dates,
            "na_values": self.na_values,
            **kwargs,
        }
        return {k: v for k, v in params.items() if v is not None}

    def _extract_impl(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {file_path}")

        self.logger.info(f"lendo CSV: {file_path.name}")
        params = self._read_params(file_path, **kwargs)

        try:
            if self.chunksize:
                chunks = list(pd.read_csv(**params, chunksize=self.chunksize))
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(**params)
        except UnicodeDecodeError:
            import chardet
            with file_path.open("rb") as f:
                detected = chardet.detect(f.read(10000))["encoding"]
            self.logger.warning(f"encoding {self.encoding} falhou, usando {detected}")
            params["encoding"] = detected
            df = pd.read_csv(**params)

        self.logger.info(f"CSV: {len(df)} linhas, {len(df.columns)} colunas")
        return df

    def extract_sample(self, file_path: Union[str, Path], n_rows: int = 5) -> pd.DataFrame:
        prev = self.cache_enabled
        self.cache_enabled = False
        try:
            return self._extract_impl(file_path, nrows=n_rows)
        finally:
            self.cache_enabled = prev
