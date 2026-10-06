"""适用于中文法律检索的 BGE 向量模型封装。"""

from __future__ import annotations

from pathlib import Path

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def resolve_device(requested: str) -> str:
    """优先使用配置设备，CUDA 不可用时安全回退到 CPU。"""

    normalized = requested.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return normalized or "cpu"


class BgeChineseEmbeddings(Embeddings):
    """文档直接编码、查询增加 BGE 中文检索指令。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_folder: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        self.device = resolve_device(device)
        self.model_name = model_name
        self._backend = HuggingFaceEmbeddings(
            model=model_name,
            cache_folder=str(cache_folder) if cache_folder else None,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
            query_encode_kwargs={"normalize_embeddings": True},
            show_progress=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._backend.embed_query(f"{BGE_QUERY_INSTRUCTION}{text.strip()}")

