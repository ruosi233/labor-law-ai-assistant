from __future__ import annotations

from labor_law_ai.agent import LaborLawAgent, classify_intent
from labor_law_ai.config import Settings
from labor_law_ai.rag import RetrievedArticle


class StubRetriever:
    def search(self, query: str):
        return [
            RetrievedArticle(
                page_content=(
                    "《中华人民共和国劳动合同法》第八十七条\n第七章 法律责任\n"
                    "用人单位违法解除或者终止劳动合同的，应当按经济补偿标准的二倍支付赔偿金。"
                ),
                score=0.93,
                metadata={
                    "law_name": "中华人民共和国劳动合同法",
                    "article_number": 87,
                    "article_label": "第八十七条",
                    "source_url": "https://www.mohrss.gov.cn/example",
                    "chunk_index": 0,
                },
            )
        ]


def offline_agent() -> LaborLawAgent:
    settings = Settings(llm_api_key="", embedding_device="cpu")
    return LaborLawAgent(settings=settings, retriever=StubRetriever(), llm=None)


def test_rule_router_has_four_core_paths() -> None:
    assert classify_intent("什么是无固定期限劳动合同？") == "direct"
    assert classify_intent("劳动合同法第八十七条怎么规定？") == "rag"
    assert classify_intent("公司辞退我能赔多少钱？") == "calculate"
    assert classify_intent("  ") == "clarify"


def test_rag_path_forces_source() -> None:
    state = offline_agent().invoke("劳动合同法第八十七条怎么规定？", "rag-test")

    assert state["action"] == "rag"
    assert state["citations_valid"] is True
    assert "第八十七条" in state["answer"]
    assert "法律依据" in state["answer"]
    assert any("RAG" in step for step in state["trace"])


def test_calculation_asks_then_resumes_with_same_thread() -> None:
    agent = offline_agent()
    first = agent.invoke("公司辞退我能赔多少钱？", "calculation-thread")

    assert first["action"] == "calculate"
    assert "为了避免误算" in first["answer"]
    assert "请按下面格式回复" in first["answer"]
    assert "入职日期：2023-09-20" in first["answer"]
    assert "解除前十二个月平均工资：8000元" in first["answer"]
    assert "start_date" in first["missing_fields"]

    second = agent.invoke(
        "入职日期：2020-01-01；解除日期：2024-01-01；"
        "平均月工资：10000元；解除原因：公司违法解除；2N",
        "calculation-thread",
    )

    assert second["missing_fields"] == []
    assert second["calculation_result"]["compensation_type"] == "2N"
    assert second["calculation_result"]["total"] == "80000.00"
    assert "第八十七条" in second["answer"]
    assert any("补偿计算" in step for step in second["trace"])


def test_direct_concept_does_not_need_rag() -> None:
    state = offline_agent().invoke("什么是经济补偿？", "direct-test")

    assert state["action"] == "direct"
    assert "工作年限" in state["answer"]
    assert state["citations_valid"] is True
