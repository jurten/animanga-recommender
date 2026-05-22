from __future__ import annotations

import json
import secrets
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, get_settings


API_BASE = "https://api.myanimelist.net/v2"
AUTH_URL = "https://myanimelist.net/v1/oauth2/authorize"
TOKEN_URL = "https://myanimelist.net/v1/oauth2/token"


ANIME_FIELDS = ",".join(
    [
        "id",
        "title",
        "main_picture",
        "alternative_titles",
        "start_date",
        "end_date",
        "synopsis",
        "mean",
        "rank",
        "popularity",
        "num_list_users",
        "num_scoring_users",
        "nsfw",
        "genres",
        "created_at",
        "updated_at",
        "media_type",
        "status",
        "num_episodes",
        "start_season",
        "source",
        "rating",
        "studios",
        "my_list_status",
    ]
)

MANGA_FIELDS = ",".join(
    [
        "id",
        "title",
        "main_picture",
        "alternative_titles",
        "start_date",
        "end_date",
        "synopsis",
        "mean",
        "rank",
        "popularity",
        "num_list_users",
        "num_scoring_users",
        "nsfw",
        "genres",
        "created_at",
        "updated_at",
        "media_type",
        "status",
        "num_volumes",
        "num_chapters",
        "authors",
        "serialization",
        "my_list_status",
    ]
)


def make_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(96)[:128]
    # MAL currently supports the plain PKCE method, so the challenge must match
    # the verifier sent to the token endpoint.
    return verifier, verifier


class TokenStore:
    def __init__(self, path: Path | None = None):
        self.path = path or get_settings().token_path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))


class MALClient:
    def __init__(self, settings: Settings | None = None, token_store: TokenStore | None = None):
        self.settings = settings or get_settings()
        self.token_store = token_store or TokenStore(self.settings.token_path)

    def auth_url(self, code_challenge: str, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.settings.mal_client_id,
            "code_challenge": code_challenge,
            "state": state,
            "redirect_uri": self.settings.mal_redirect_uri,
        }
        return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str, code_verifier: str) -> dict[str, Any]:
        data = {
            "client_id": self.settings.mal_client_id,
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": self.settings.mal_redirect_uri,
        }
        if self.settings.mal_client_secret:
            data["client_secret"] = self.settings.mal_client_secret
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(TOKEN_URL, data=data)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"MAL token exchange failed: {response.text}") from exc
        tokens = response.json()
        self.token_store.save(tokens)
        return tokens

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        tokens = self.token_store.load()
        access_token = tokens.get("access_token")
        if not access_token:
            raise RuntimeError("Missing MAL access token. Run `mal-rec mal-auth login` first.")
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {access_token}"
        async with httpx.AsyncClient(base_url=API_BASE, timeout=60) as client:
            response = await client.request(method, path, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()

    async def get_user(self) -> dict[str, Any]:
        return await self.request("GET", "/users/@me", params={"fields": "anime_statistics"})

    async def iter_user_list(self, content_type: str, limit: int = 100):
        fields = ANIME_FIELDS if content_type == "anime" else MANGA_FIELDS
        path = f"/users/@me/{content_type}list"
        params: dict[str, Any] = {"fields": fields, "limit": limit, "nsfw": "true"}
        while True:
            payload = await self.request("GET", path, params=params)
            for edge in payload.get("data", []):
                yield edge
            next_url = payload.get("paging", {}).get("next")
            if not next_url:
                break
            parsed = urllib.parse.urlparse(next_url)
            path = parsed.path.removeprefix("/v2")
            params = dict(urllib.parse.parse_qsl(parsed.query))

    async def get_item(self, content_type: str, mal_id: int) -> dict[str, Any]:
        fields = ANIME_FIELDS if content_type == "anime" else MANGA_FIELDS
        return await self.request("GET", f"/{content_type}/{mal_id}", params={"fields": fields})
