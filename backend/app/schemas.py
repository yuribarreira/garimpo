from typing import Any, Dict, List

import pandas as pd


class SchemaError(ValueError):
    pass


class SchemaValidator:
    def validate(self, data: Any, schema: Dict[str, Any]) -> bool:
        required: List[str] = schema.get("required", [])
        if not required:
            return True

        if isinstance(data, pd.DataFrame):
            cols = set(data.columns)
        elif isinstance(data, dict):
            cols = set(data.keys())
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            cols = set(data[0].keys())
        else:
            cols = set()

        missing = [c for c in required if c not in cols]
        if missing:
            raise SchemaError(f"campos obrigatórios ausentes: {missing}")
        return True
