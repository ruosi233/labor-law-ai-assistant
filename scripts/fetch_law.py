"""下载并解析人社部发布的《中华人民共和国劳动合同法》。"""

from __future__ import annotations

import json
from pathlib import Path

from labor_law_ai.law_loader import (
    article_summary,
    download_official_html,
    save_law_dataset,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    html = download_official_html()
    articles = save_law_dataset(
        html=html,
        raw_html_path=root / "data" / "raw" / "labor_contract_law.html",
        clean_text_path=root / "data" / "processed" / "labor_contract_law.txt",
        json_path=root / "data" / "processed" / "labor_contract_law.json",
    )
    print(json.dumps(article_summary(articles), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

