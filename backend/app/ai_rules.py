import ast
import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o"

ALLOWED_IMPORTS = {"pandas", "numpy", "re", "json", "math", "datetime", "logging"}
BANNED_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}


class UnsafeCodeError(ValueError):
    pass


class CodeChecker(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                self.errors.append(f"import proibido: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            self.errors.append(f"import proibido: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            self.errors.append(f"chamada proibida: {node.func.id}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"acesso a dunder bloqueado: {node.attr}")
        self.generic_visit(node)


def check_code(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"is_valid": False, "errors": [f"erro de sintaxe: {e}"], "warnings": []}

    checker = CodeChecker()
    checker.visit(tree)
    warnings = []
    if "def transform_data" not in code:
        warnings.append("função transform_data não encontrada")
    return {"is_valid": not checker.errors, "errors": checker.errors, "warnings": warnings}


def _safe_exec(code: str) -> dict:
    safe_builtins = {
        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "sorted": sorted, "map": map, "filter": filter, "any": any, "all": all,
        "isinstance": isinstance, "print": print,
    }
    ns = {"__builtins__": safe_builtins, "pd": pd, "np": np}
    exec(code, ns)
    return ns


PROMPT = """Você é um engenheiro de dados. Escreva uma função Python que transforma um DataFrame pandas.

Requisito: "{requirement}"
{ctx}
Regras:
- assinatura: def transform_data(df: pd.DataFrame) -> pd.DataFrame
- use apenas pandas e numpy
- retorne sempre um DataFrame
- responda só com o bloco de código

```python
def transform_data(df):
    ...
```"""


class AIRuleGenerator:
    def __init__(self, claude_key: Optional[str] = None, openai_key: Optional[str] = None,
                 prefer: str = "claude"):
        self.prefer = prefer
        self.claude = None
        self.openai = None
        self.cache: dict = {}

        ckey = claude_key or os.getenv("ANTHROPIC_API_KEY")
        if ckey and HAS_ANTHROPIC:
            self.claude = anthropic.Anthropic(api_key=ckey)

        okey = openai_key or os.getenv("OPENAI_API_KEY")
        if okey and HAS_OPENAI:
            self.openai = OpenAI(api_key=okey)

        if not self.claude and not self.openai:
            raise RuntimeError("nenhuma API de IA configurada (ANTHROPIC_API_KEY ou OPENAI_API_KEY)")

    def generate_transformation_rule(self, requirement: str,
                                     data_context: Optional[Dict] = None) -> Dict[str, Any]:
        key = requirement.lower().strip()
        if key in self.cache:
            return self.cache[key]

        ctx = ""
        if data_context and data_context.get("columns"):
            ctx = f"Colunas disponíveis: {data_context['columns']}\n"
        prompt = PROMPT.format(requirement=requirement, ctx=ctx)

        started = datetime.now()
        raw = self._call_ai(prompt)
        code = self._extract_code(raw)
        if not code:
            raise ValueError("a IA não retornou código Python válido")

        validation = check_code(code)
        test = self._smoke_test(code) if validation["is_valid"] else None

        result = {
            "requirement": requirement,
            "generated_code": code,
            "validation": validation,
            "test_result": test,
            "metadata": {
                "model": self.prefer,
                "generation_time": (datetime.now() - started).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            },
        }
        self.cache[key] = result
        return result

    def _call_ai(self, prompt: str) -> str:
        order = [self.prefer, "openai" if self.prefer == "claude" else "claude"]
        for which in order:
            if which == "claude" and self.claude:
                res = self.claude.messages.create(
                    model=CLAUDE_MODEL, max_tokens=2000, temperature=0.2,
                    messages=[{"role": "user", "content": prompt}])
                return res.content[0].text
            if which == "openai" and self.openai:
                res = self.openai.chat.completions.create(
                    model=OPENAI_MODEL, temperature=0.2,
                    messages=[{"role": "user", "content": prompt}])
                return res.choices[0].message.content
        raise RuntimeError("nenhuma API de IA disponível")

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
        if "def transform_data" in text:
            return text[text.find("def transform_data"):].strip()
        return None

    def _smoke_test(self, code: str) -> Dict[str, Any]:
        sample = pd.DataFrame({
            "nome": ["João Silva", "Maria Santos"],
            "email": ["joao@x.com", "maria@y.com"],
            "valor": [500, 1500],
        })
        try:
            ns = _safe_exec(code)
            fn = ns.get("transform_data")
            if not callable(fn):
                return {"success": False, "error": "transform_data não definida"}
            out = fn(sample.copy())
            if not isinstance(out, pd.DataFrame):
                return {"success": False, "error": "retorno não é DataFrame"}
            return {"success": True, "input_shape": sample.shape, "output_shape": out.shape}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def execute_rule(self, rule: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        if not rule["validation"]["is_valid"]:
            raise UnsafeCodeError(f"regra inválida: {rule['validation']['errors']}")
        ns = _safe_exec(rule["generated_code"])
        fn = ns.get("transform_data")
        if not callable(fn):
            raise ValueError("transform_data não encontrada no código gerado")
        return fn(df)
