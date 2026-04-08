from abc import abstractmethod
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import logging

from ..base import ExtractBase
from ..schemas import SchemaValidator


class BaseExtractor(ExtractBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.cache_dir = Path(self.config.get("cache_dir", ".cache/extractors"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.schema = self.config.get("schema")
        self.validator = SchemaValidator() if self.schema else None

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        ))
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, self.config.get("log_level", "INFO").upper()))

    def _cache_key(self, params: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()

    def _read_cache(self, key: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            self.logger.warning(f"cache ilegível ({key}): {e}")
            return None
        if datetime.now() > cached["expiry"]:
            cache_file.unlink(missing_ok=True)
            return None
        return cached["data"]

    def _write_cache(self, key: str, data: Any) -> None:
        if not self.cache_enabled:
            return
        cache_file = self.cache_dir / f"{key}.pkl"
        payload = {"data": data, "expiry": datetime.now() + timedelta(seconds=self.cache_ttl)}
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)

    def validate_data(self, data: Any) -> bool:
        if not self.validator or not self.schema:
            return True
        return self.validator.validate(data, self.schema)

    def extract(self, **kwargs) -> Any:
        key = self._cache_key(kwargs)
        cached = self._read_cache(key)
        if cached is not None:
            self.logger.info("usando dados do cache")
            return cached

        self.logger.info(f"extraindo via {self.__class__.__name__}")
        data = self._extract_impl(**kwargs)
        if self.schema:
            self.validate_data(data)
        self._write_cache(key, data)
        return data

    @abstractmethod
    def _extract_impl(self, **kwargs) -> Any:
        ...

    def clear_cache(self) -> None:
        if not self.cache_dir.exists():
            return
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
