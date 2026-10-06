import json
from pathlib import Path


DATASET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "labor_contract_law.json"
)


def test_official_dataset_is_complete_and_sequential() -> None:
    articles = json.loads(DATASET.read_text(encoding="utf-8"))

    assert len(articles) == 98
    assert [article["article_number"] for article in articles] == list(range(1, 99))
    assert all(article["law_name"] == "中华人民共和国劳动合同法" for article in articles)
    assert all(article["source_url"].startswith("https://www.mohrss.gov.cn/") for article in articles)


def test_key_compensation_articles_are_present() -> None:
    articles = {
        article["article_number"]: article
        for article in json.loads(DATASET.read_text(encoding="utf-8"))
    }

    assert "额外支付劳动者一个月工资" in articles[40]["content"]
    assert "每满一年支付一个月工资" in articles[47]["content"]
    assert "二倍向劳动者支付赔偿金" in articles[87]["content"]

