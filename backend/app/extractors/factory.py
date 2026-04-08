from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from .base import BaseExtractor
from .csv_extractor import CSVExtractor
from .excel_extractor import ExcelExtractor
from .api_extractor import APIExtractor
from .sqlite_extractor import SQLiteExtractor
from .json_extractor import JSONExtractor


class ExtractorFactory:
    _extractors: Dict[str, Type[BaseExtractor]] = {
        "csv": CSVExtractor,
        "excel": ExcelExtractor,
        "xlsx": ExcelExtractor,
        "xls": ExcelExtractor,
        "api": APIExtractor,
        "rest": APIExtractor,
        "sqlite": SQLiteExtractor,
        "sqlite3": SQLiteExtractor,
        "db": SQLiteExtractor,
        "json": JSONExtractor,
        "jsonl": JSONExtractor,
    }

    _ext_map = {
        "csv": "csv", "tsv": "csv", "txt": "csv",
        "xlsx": "excel", "xls": "excel", "xlsm": "excel", "xlsb": "excel",
        "json": "json", "jsonl": "jsonl",
        "db": "sqlite", "sqlite": "sqlite", "sqlite3": "sqlite",
    }

    @classmethod
    def create(cls, source_type: str, config: Optional[Dict[str, Any]] = None) -> BaseExtractor:
        source_type = source_type.lower()
        if source_type not in cls._extractors:
            raise ValueError(
                f"fonte '{source_type}' não suportada. disponíveis: {list(cls._extractors)}"
            )
        return cls._extractors[source_type](config)

    @classmethod
    def create_from_file(cls, file_path: Union[str, Path],
                         config: Optional[Dict[str, Any]] = None) -> BaseExtractor:
        ext = Path(file_path).suffix.lower().lstrip(".")
        if not ext:
            raise ValueError(f"arquivo sem extensão: {file_path}")
        if ext not in cls._ext_map:
            raise ValueError(f"extensão '{ext}' não suportada. disponíveis: {list(cls._ext_map)}")

        config = config or {}
        if ext == "tsv":
            config.setdefault("delimiter", "\t")
        elif ext == "jsonl":
            config["json_lines"] = True
        return cls.create(cls._ext_map[ext], config)

    @classmethod
    def register(cls, source_type: str, extractor_cls: Type[BaseExtractor]) -> None:
        if not issubclass(extractor_cls, BaseExtractor):
            raise ValueError("extractor deve herdar de BaseExtractor")
        cls._extractors[source_type.lower()] = extractor_cls

    @classmethod
    def list_types(cls) -> list:
        return list(cls._extractors)
