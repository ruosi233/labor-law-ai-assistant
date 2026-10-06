"""《劳动合同法》文档切分、Chroma 建库与检索。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from labor_law_ai.config import PROJECT_ROOT, Settings, get_settings
from labor_law_ai.embeddings import BgeChineseEmbeddings
from labor_law_ai.law_loader import chinese_number_to_int


COLLECTION_NAME = "labor_contract_law"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "labor_contract_law.json"
_ARTICLE_REFERENCE_RE = re.compile(r"第\s*([零〇一二三四五六七八九十百\d]+)\s*条")


def _cosine_relevance(distance: float) -> float:
    """将 Chroma 余弦距离转成稳定在 [0, 1] 的相关度。"""

    return max(0.0, min(1.0, 1.0 - float(distance)))


@dataclass(frozen=True)
class RetrievedArticle:
    page_content: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _referenced_article_number(query: str) -> int | None:
    match = _ARTICLE_REFERENCE_RE.search(query)
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else chinese_number_to_int(value)


def load_article_documents(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    chunk_size: int = 700,
    chunk_overlap: int = 80,
) -> list[Document]:
    """以条文为边界，超长条文再按中文语义标点切分。"""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    articles = json.loads(dataset_path.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "；", "。", "，", ""],
        length_function=len,
    )
    documents: list[Document] = []

    for article in articles:
        pieces = splitter.split_text(article["content"])
        for chunk_index, piece in enumerate(pieces):
            metadata = {
                "law_name": article["law_name"],
                "article_number": article["article_number"],
                "article_label": article["article_label"],
                "chapter": article["chapter"],
                "section": article["section"] or "",
                "version": article["version"],
                "source_url": article["source_url"],
                "issuing_authority": article["issuing_authority"],
                "source_publisher": article["source_publisher"],
                "chunk_index": chunk_index,
                "chunk_count": len(pieces),
            }
            page_content = (
                f"《{article['law_name']}》{article['article_label']}\n"
                f"{article['chapter']}\n{piece}"
            )
            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def document_id(document: Document) -> str:
    return (
        "labor-contract-law-"
        f"article-{int(document.metadata['article_number']):03d}-"
        f"chunk-{int(document.metadata['chunk_index']):02d}"
    )


class LaborLawRetriever:
    """支持条文号精确召回和向量相似度过滤的检索器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        vector_store: Chroma | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embeddings = embeddings
        self._vector_store = vector_store

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            model_name = self.settings.embedding_model
            local_model = Path(model_name)
            if not local_model.is_absolute():
                project_model = (PROJECT_ROOT / local_model).resolve()
                if project_model.exists():
                    model_name = str(project_model)
            self._embeddings = BgeChineseEmbeddings(
                model_name=model_name,
                device=self.settings.embedding_device,
                cache_folder=self.settings.model_cache_dir,
            )
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.settings.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"},
                relevance_score_fn=_cosine_relevance,
            )
        return self._vector_store

    def count(self) -> int:
        return self.vector_store._collection.count()

    def build_index(self, documents: list[Document], reset: bool = False) -> int:
        store = self.vector_store
        if reset:
            store.reset_collection()
        ids = [document_id(document) for document in documents]
        if store._collection.count():
            existing = set(store.get(ids=ids, include=[]).get("ids", []))
        else:
            existing = set()
        new_documents = [doc for doc, doc_id in zip(documents, ids) if doc_id not in existing]
        new_ids = [doc_id for doc_id in ids if doc_id not in existing]
        if new_documents:
            store.add_documents(documents=new_documents, ids=new_ids)
        return store._collection.count()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedArticle]:
        query = query.strip()
        if not query:
            return []

        k = top_k or self.settings.retrieval_top_k
        threshold = (
            self.settings.retrieval_score_threshold
            if score_threshold is None
            else score_threshold
        )
        results: list[RetrievedArticle] = []
        seen: set[tuple[int, int]] = set()

        article_number = _referenced_article_number(query)
        if article_number is not None:
            exact = self.vector_store.get(
                where={"article_number": article_number},
                include=["documents", "metadatas"],
            )
            for content, metadata in zip(
                exact.get("documents", []), exact.get("metadatas", [])
            ):
                key = (int(metadata["article_number"]), int(metadata["chunk_index"]))
                seen.add(key)
                results.append(
                    RetrievedArticle(page_content=content, score=1.0, metadata=metadata)
                )

        semantic = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        for document, score in semantic:
            normalized_score = max(0.0, min(1.0, float(score)))
            key = (
                int(document.metadata["article_number"]),
                int(document.metadata["chunk_index"]),
            )
            if normalized_score < threshold or key in seen:
                continue
            seen.add(key)
            results.append(
                RetrievedArticle(
                    page_content=document.page_content,
                    score=normalized_score,
                    metadata=dict(document.metadata),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:k]
