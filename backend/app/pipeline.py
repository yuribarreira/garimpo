import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from time import sleep

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .brazilian_utils import BrazilianDataValidator
except ImportError:
    from brazilian_utils import BrazilianDataValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CSV_ENCODINGS = ["utf-8", "iso-8859-1", "cp1252", "latin-1"]
CSV_SEPS = [",", ";", "|", "\t"]
POSITIVOS = {"alta", "sim", "1", "high"}


class LeadPipeline:
    def __init__(self, model_dir: str = "./production_models/", enable_api_calls: bool = False):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.enable_api_calls = enable_api_calls
        self.cnpj_cache: dict = {}
        self.model = None
        self.metadata: dict = {}
        self.target_column = "qualidade_lead"

    def _load_data(self, file_path: str) -> pd.DataFrame:
        ext = Path(file_path).suffix.lower()
        logger.info(f"carregando {file_path}")
        if ext == ".csv":
            for enc in CSV_ENCODINGS:
                for sep in CSV_SEPS:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=enc, low_memory=False)
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                    if len(df.columns) > 1:
                        logger.info(f"CSV lido (sep='{sep}', enc='{enc}')")
                        return df
            raise ValueError("não consegui ler o CSV: verifique separador e encoding")
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        raise ValueError(f"formato não suportado: {ext}")

    def _dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        doc_col = next((c for c in df.columns if "CPF" in c.upper()), None) \
            or next((c for c in df.columns if "CNPJ" in c.upper()), None)
        if not doc_col:
            logger.warning("sem coluna CPF/CNPJ, pulando deduplicação")
            return df
        date_col = next((c for c in df.columns if "DATE" in c.upper() or "DATA" in c.upper()), None)
        df = df.dropna(subset=[doc_col])
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.sort_values(by=date_col, ascending=False)
        before = len(df)
        df = df.drop_duplicates(subset=[doc_col], keep="first")
        logger.info(f"{before - len(df)} duplicados removidos por '{doc_col}'")
        return df

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("").astype(str)

        if "nome" in df.columns:
            df["nome"] = df["nome"].apply(BrazilianDataValidator.normalize_name)
        df["nome_length"] = df["nome"].str.len() if "nome" in df.columns else 0

        empresa_col = "empresa_cargo" if "empresa_cargo" in df.columns else (
            "empresa" if "empresa" in df.columns else None)
        df["empresa_cargo_length"] = df[empresa_col].str.len() if empresa_col else 0

        df["has_linkedin"] = (
            df["linkedin"].str.contains("linkedin", case=False).astype(int)
            if "linkedin" in df.columns else 0)
        df["has_email"] = (
            df["email"].str.contains("@").astype(int) if "email" in df.columns else 0)

        phone_col = "whatsapp" if "whatsapp" in df.columns else (
            "telefone" if "telefone" in df.columns else None)
        df["has_whatsapp"] = (
            df[phone_col].apply(BrazilianDataValidator.validate_phone_whatsapp).astype(int)
            if phone_col else 0)

        cpf_col = next((c for c in df.columns if "CPF" in c.upper()), None)
        cnpj_col = next((c for c in df.columns if "CNPJ" in c.upper()), None)
        df["cpf_valido"] = (
            df[cpf_col].apply(BrazilianDataValidator.validate_cpf).astype(int) if cpf_col else 0)
        df["cnpj_valido_formato"] = (
            df[cnpj_col].apply(BrazilianDataValidator.validate_cnpj).astype(int) if cnpj_col else 0)

        df = BrazilianDataValidator.extract_phone_features(df)

        df["data_quality_score"] = (
            df["has_email"] + df["has_linkedin"] + df["has_whatsapp"]
            + df["cpf_valido"] + df["cnpj_valido_formato"]
            + (df["nome_length"] > 5).astype(int)
        )
        return df

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enable_api_calls:
            return df
        cnpj_col = next((c for c in df.columns if "CNPJ" in c.upper()), None)
        if not cnpj_col:
            logger.warning("sem coluna CNPJ para enriquecer")
            return df

        df["empresa_porte"] = "N/A"
        for idx, row in df.iterrows():
            raw = row.get(cnpj_col)
            if pd.isna(raw) or not str(raw).strip():
                continue
            cnpj = re.sub(r"\D", "", str(raw))
            info = self.cnpj_cache.get(cnpj)
            if info is None:
                sleep(0.5)
                try:
                    res = requests.get(
                        f"https://www.receitaws.com.br/v1/cnpj/{cnpj}", timeout=10)
                    info = res.json() if res.ok and res.json().get("status") != "ERROR" else {}
                except (requests.RequestException, ValueError) as e:
                    logger.warning(f"falha ao consultar CNPJ {cnpj}: {e}")
                    info = {}
                self.cnpj_cache[cnpj] = info
            if info:
                df.loc[idx, "empresa_porte"] = info.get("porte", "N/A")
        return df

    def run_etl(self, file_path: str) -> pd.DataFrame:
        df = self._load_data(file_path)
        df = self._dedup(df)
        df = self._features(df)
        df = self._enrich(df)
        logger.info("ETL concluído")
        return df

    def train_model(self, df: pd.DataFrame, target_column: str = "qualidade_lead") -> None:
        self.target_column = target_column
        if target_column not in df.columns:
            raise ValueError(f"coluna alvo '{target_column}' não encontrada")

        y = df[target_column].apply(lambda x: 1 if str(x).lower() in POSITIVOS else 0)
        X = df.drop(columns=[target_column])

        num = X.select_dtypes(include=np.number).columns.tolist()
        cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
        pre = ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
        self.model = Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ])

        n, n_classes = len(X), y.nunique()
        if n < 10:
            logger.warning(f"dataset pequeno ({n} amostras), treino == teste")
            X_tr = X_te = X
            y_tr = y_te = y
        else:
            test_size = min(0.8, max(0.2, 2 * n_classes / n))
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if n >= 2 * n_classes else None)

        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)
        logger.info("\n" + classification_report(y_te, y_pred))
        try:
            auc = roc_auc_score(y_te, self.model.predict_proba(X_te)[:, 1])
            logger.info(f"AUC: {auc:.4f}")
        except ValueError as e:
            logger.warning(f"AUC indisponível: {e}")

        self.metadata = {
            "training_date": datetime.now().isoformat(),
            "model_type": "RandomForestClassifier",
            "numeric_features": num,
            "categorical_features": cat,
            "target_column": target_column,
        }
        joblib.dump(self.model, self.model_dir / "model_pipeline.joblib")
        (self.model_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"modelo salvo em {self.model_dir}")

    def _load_model(self) -> None:
        path = self.model_dir / "model_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError("nenhum modelo treinado, rode o modo 'train' primeiro")
        self.model = joblib.load(path)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self._load_model()
        out = df.copy()
        preds = self.model.predict(df)
        out["qualidade_predita"] = np.where(preds == 1, "Alta", "Baixa")
        out["confianca_predicao"] = self.model.predict_proba(df)[:, 1]
        return out


def _save(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False, sep=";", encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="Garimpo - pipeline ETL + qualificação de leads")
    ap.add_argument("--mode", required=True, choices=["train", "predict", "etl"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--model-dir", default="./production_models/")
    ap.add_argument("--target", default="qualidade_lead")
    ap.add_argument("--enable-api", action="store_true")
    args = ap.parse_args()

    pipe = LeadPipeline(model_dir=args.model_dir, enable_api_calls=args.enable_api)
    df = pipe.run_etl(args.input)

    if args.mode == "train":
        pipe.train_model(df, target_column=args.target)
    elif args.mode in ("predict", "etl"):
        if not args.output:
            raise ValueError(f"--output é obrigatório no modo '{args.mode}'")
        result = pipe.batch_predict(df) if args.mode == "predict" else df
        _save(result, args.output)
        logger.info(f"resultado salvo em {args.output}")


if __name__ == "__main__":
    main()
