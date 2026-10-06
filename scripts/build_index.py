"""构建并验证《劳动合同法》Chroma 向量索引。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from labor_law_ai.config import get_settings
from labor_law_ai.rag import LaborLawRetriever, load_article_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建劳动合同法 Chroma 索引")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空原集合后重新构建",
    )
    parser.add_argument(
        "--query",
        default="公司违法解除劳动合同，应当如何支付赔偿金？",
        help="建库后用于验证检索结果的问题",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    documents = load_article_documents(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"向量模型：{settings.embedding_model}")
    print(f"请求设备：{settings.embedding_device}")
    print(f"模型缓存：{settings.model_cache_dir}")
    print(f"索引目录：{settings.chroma_dir}")
    print(f"待索引文档：{len(documents)}")

    retriever = LaborLawRetriever(settings=settings)
    count = retriever.build_index(documents, reset=args.reset)
    print(f"实际设备：{retriever.embeddings.device}")
    print(f"索引记录：{count}")

    results = retriever.search(args.query)
    print(f"验证问题：{args.query}")
    if not results:
        raise RuntimeError("索引已构建，但验证检索没有返回通过阈值的法条")

    for position, item in enumerate(results, start=1):
        label = item.metadata["article_label"]
        law_name = item.metadata["law_name"]
        print(f"{position}. 《{law_name}》{label}，相关度 {item.score:.4f}")
        print(f"   {item.page_content.replace(chr(10), ' ')[:180]}")


if __name__ == "__main__":
    main()
