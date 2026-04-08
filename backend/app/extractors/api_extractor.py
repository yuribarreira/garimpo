import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import BaseExtractor


class APIExtractor(BaseExtractor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = self.config.get("base_url", "")
        self.headers = self.config.get("headers", {})
        self.auth = self.config.get("auth")
        self.timeout = self.config.get("timeout", 30)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 0)
        self.max_retries = self.config.get("max_retries", 3)
        self.backoff_factor = self.config.get("backoff_factor", 2)
        self.retry_statuses = self.config.get("retry_statuses", [429, 500, 502, 503, 504])
        self.session = self._make_session()
        self._last_req = None

    def _make_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            status_forcelist=self.retry_statuses,
            backoff_factor=self.backoff_factor,
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)

        if isinstance(self.auth, dict):
            if "bearer" in self.auth:
                session.headers["Authorization"] = f"Bearer {self.auth['bearer']}"
            elif "api_key" in self.auth:
                session.headers["X-API-Key"] = self.auth["api_key"]
            elif "username" in self.auth and "password" in self.auth:
                session.auth = (self.auth["username"], self.auth["password"])
        return session

    def _throttle(self) -> None:
        if self.rate_limit_delay > 0 and self._last_req:
            elapsed = time.time() - self._last_req
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        if not url.startswith(("http://", "https://")):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        self._throttle()
        kwargs.setdefault("timeout", self.timeout)

        self.logger.info(f"{method} {url}")
        res = self.session.request(method, url, **kwargs)
        res.raise_for_status()
        self._last_req = time.time()
        return res

    def _extract_impl(self, endpoint: str = "", method: str = "GET", **kwargs) -> Any:
        res = self._request(method, endpoint, **kwargs)
        if "application/json" in res.headers.get("Content-Type", ""):
            data = res.json()
        else:
            data = res.text
        if isinstance(data, dict):
            data["_metadata"] = {
                "status_code": res.status_code,
                "timestamp": datetime.now().isoformat(),
                "url": res.url,
            }
        return data

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        return self.extract(endpoint=endpoint, method="GET", params=params, **kwargs)

    def post(self, endpoint: str, data=None, json=None, **kwargs) -> Any:
        return self.extract(endpoint=endpoint, method="POST", data=data, json=json, **kwargs)

    def paginate(self, endpoint: str, page_param: str = "page", per_page: int = 100,
                 max_pages: Optional[int] = None, response_key: Optional[str] = None) -> List[Any]:
        out: List[Any] = []
        page = 1
        while True:
            if max_pages and page > max_pages:
                break
            res = self.get(endpoint, params={page_param: page, "per_page": per_page})
            page_data = res.get(response_key, []) if (response_key and isinstance(res, dict)) else res
            if not page_data:
                break
            out.extend(page_data if isinstance(page_data, list) else [page_data])
            self.logger.info(f"página {page}: {len(page_data)} itens")
            page += 1
        return out

    def health_check(self, endpoint: str = "/health") -> bool:
        try:
            return self._request("GET", endpoint).status_code in (200, 204)
        except requests.RequestException as e:
            self.logger.error(f"health check falhou: {e}")
            return False
