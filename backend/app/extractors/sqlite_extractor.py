import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .base import BaseExtractor


class SQLiteExtractor(BaseExtractor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.query = self.config.get("query")
        self.table = self.config.get("table")

    def _extract_impl(self, db_path: Union[str, Path], query: Optional[str] = None,
                      table: Optional[str] = None, **kwargs) -> pd.DataFrame:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"banco SQLite não encontrado: {db_path}")

        query = query or self.query
        table = table or self.table
        if not query and not table:
            raise ValueError("informe 'query' ou 'table' para extrair do SQLite")

        if not query:
            query = f"SELECT * FROM {table}"

        self.logger.info(f"Lendo SQLite: {db_path.name}")
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(query, conn)

        self.logger.info(f"SQLite extraído: {len(df)} linhas, {len(df.columns)} colunas")
        return df

    def list_tables(self, db_path: Union[str, Path]) -> list:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"banco SQLite não encontrado: {db_path}")
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]
