import logging
from typing import Any, Dict, Optional


class ExtractBase:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, **kwargs) -> Any:
        raise NotImplementedError("cada extractor deve implementar extract()")
