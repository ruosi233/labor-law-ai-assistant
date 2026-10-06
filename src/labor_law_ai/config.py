"""应用配置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """从项目根目录的 .env 读取配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "gpt-4.1-mini"

    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_device: str = "cuda"
    model_cache_dir: Path = PROJECT_ROOT / "models"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    chunk_size: int = Field(default=700, ge=200, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)

    @field_validator("model_cache_dir", "chroma_dir", mode="after")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        """让 .env 中的相对目录始终相对于项目根目录。"""

        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def get_settings() -> Settings:
    return Settings()
