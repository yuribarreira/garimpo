import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


class LeadPredictor:
    def __init__(self, model_dir: str = "./production_models/"):
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "model_pipeline.joblib"
        meta_path = self.model_dir / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"modelo não encontrado em {self.model_dir}")
        self.model = joblib.load(model_path)
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        preds = self.model.predict(df)
        proba = self.model.predict_proba(df)
        out = []
        for i, (p, pr) in enumerate(zip(preds, proba)):
            out.append({
                "index": i,
                "qualidade": "Alta" if p == 1 else "Baixa",
                "confianca": float(np.max(pr)),
                "prob_alta": float(pr[1]),
                "prob_baixa": float(pr[0]),
            })
        return out
