from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

from labor_law_ai.config import Settings
from labor_law_ai.rag import LaborLawRetriever, document_id, load_article_documents


class DeterministicEmbeddings(Embeddings):
    keywords = ["合同", "解除", "补偿", "赔偿", "工资", "试用期", "派遣"]

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        values = [float(text.count(keyword)) for keyword in cls.keywords]
        values.append(1.0)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_article_first_chunking_preserves_metadata() -> None:
    documents = load_article_documents(chunk_size=700, chunk_overlap=80)

    assert len(documents) >= 98
    assert {document.metadata["article_number"] for document in documents} == set(
        range(1, 99)
    )
    first = documents[0]
    assert first.metadata["law_name"] == "中华人民共和国劳动合同法"
    assert first.metadata["article_label"] == "第一条"
    assert document_id(first) == "labor-contract-law-article-001-chunk-00"


def test_chroma_index_is_idempotent_and_exact_article_wins(tmp_path) -> None:
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        model_cache_dir=tmp_path / "models",
        embedding_device="cpu",
        retrieval_top_k=5,
        retrieval_score_threshold=0.0,
    )
    retriever = LaborLawRetriever(
        settings=settings,
        embeddings=DeterministicEmbeddings(),
    )
    documents = load_article_documents(chunk_size=700, chunk_overlap=80)

    first_count = retriever.build_index(documents, reset=True)
    second_count = retriever.build_index(documents)
    results = retriever.search("劳动合同法第八十七条如何规定？")

    assert first_count == len(documents)
    assert second_count == first_count
    assert results[0].metadata["article_number"] == 87
    assert results[0].score == 1.0
    assert "二倍向劳动者支付赔偿金" in results[0].page_content

