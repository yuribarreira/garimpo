import re
import locale
from typing import Union

import pandas as pd

UF = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
    "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

PREPOSICOES = {"de", "da", "do", "das", "dos", "e", "em", "na", "no", "para", "por"}


def _digits(v) -> str:
    return re.sub(r"\D", "", str(v))


class BrazilianDataValidator:
    @staticmethod
    def validate_phone_whatsapp(phone) -> bool:
        if phone is None or pd.isna(phone):
            return False
        d = _digits(phone)
        if d.startswith("55") and len(d) in (12, 13):
            d = d[2:]
        if len(d) not in (10, 11):
            return False
        ddd = int(d[:2])
        if ddd < 11 or ddd > 99:
            return False
        return not (len(d) == 11 and d[2] != "9")

    @staticmethod
    def format_phone_whatsapp(phone) -> str:
        if phone is None or pd.isna(phone):
            return ""
        d = _digits(phone)
        if len(d) == 13 and d.startswith("55"):
            country, area, num = d[:2], d[2:4], d[4:]
        elif len(d) in (10, 11):
            country, area, num = "55", d[:2], d[2:]
        else:
            return str(phone)
        if len(num) == 9:
            num = f"{num[:5]}-{num[5:]}"
        elif len(num) == 8:
            num = f"{num[:4]}-{num[4:]}"
        else:
            return str(phone)
        return f"+{country} ({area}) {num}"

    @staticmethod
    def validate_cpf(cpf) -> bool:
        if cpf is None or pd.isna(cpf):
            return False
        c = _digits(cpf)
        if len(c) != 11 or len(set(c)) == 1:
            return False
        for size in (9, 10):
            total = sum(int(c[i]) * (size + 1 - i) for i in range(size))
            check = (total * 10) % 11 % 10
            if check != int(c[size]):
                return False
        return True

    @staticmethod
    def validate_cnpj(cnpj) -> bool:
        if cnpj is None or pd.isna(cnpj):
            return False
        c = _digits(cnpj)
        if len(c) != 14 or len(set(c)) == 1:
            return False
        w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        w2 = [6] + w1
        for weights, pos in ((w1, 12), (w2, 13)):
            total = sum(int(d) * w for d, w in zip(c[:pos], weights))
            rest = total % 11
            check = 0 if rest < 2 else 11 - rest
            if check != int(c[pos]):
                return False
        return True

    @staticmethod
    def format_currency_brl(value: Union[float, int, str], symbol: bool = True) -> str:
        if value is None or pd.isna(value) or value == "":
            return "R$ 0,00" if symbol else "0,00"
        try:
            if isinstance(value, str):
                num = float(re.sub(r"[^\d,.-]", "", value).replace(",", "."))
            else:
                num = float(value)
        except (ValueError, TypeError):
            return "R$ 0,00" if symbol else "0,00"
        try:
            locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
            fmt = locale.currency(num, grouping=True, symbol=False)
        except (locale.Error, AttributeError):
            fmt = f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {fmt}" if symbol else fmt

    @staticmethod
    def normalize_name(name) -> str:
        if name is None or pd.isna(name):
            return ""
        words = re.sub(r"\s+", " ", str(name).strip()).lower().split()
        return " ".join(
            w if (i > 0 and w in PREPOSICOES) else w.capitalize()
            for i, w in enumerate(words)
        )

    @staticmethod
    def extract_uf(cidade_estado) -> str:
        if cidade_estado is None or pd.isna(cidade_estado):
            return ""
        text = str(cidade_estado).strip()
        for sep in ("/", "-", ","):
            parts = text.split(sep)
            if len(parts) >= 2 and parts[-1].strip().upper() in UF:
                return parts[-1].strip().upper()
        last = text.split()[-1].upper() if text.split() else ""
        return last if last in UF else ""

    @staticmethod
    def extract_phone_features(df: pd.DataFrame, phone_cols=None) -> pd.DataFrame:
        if phone_cols is None:
            phone_cols = [c for c in df.columns
                          if any(k in c.lower() for k in ("telefone", "whatsapp", "phone", "fone"))]
        out = df.copy()
        for col in phone_cols:
            if col not in df.columns:
                continue
            name = col.replace("telefone", "phone").replace("whatsapp", "whats")
            out[f"{name}_valido"] = df[col].apply(
                BrazilianDataValidator.validate_phone_whatsapp).astype(int)
            out[f"{name}_celular"] = df[col].apply(
                lambda x: int(bool(x) and len(_digits(x)) == 11 and _digits(x)[2] == "9"))
            out[f"{name}_tem_ddi"] = df[col].apply(
                lambda x: int(bool(x) and _digits(x).startswith("55")))
        return out
