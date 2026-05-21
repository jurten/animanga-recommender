from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mal_client_id: str = ""
    mal_redirect_uri: str = "http://localhost:8765/callback"
    database_url: str = "sqlite:///./data/recommender.db"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    prompt_version: str = "traits-v1"
    token_path: Path = Field(default=Path("./data/mal_tokens.json"))

    @property
    def sqlite_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// DATABASE_URL values are supported in v1")
        path = Path(self.database_url.removeprefix(prefix))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
