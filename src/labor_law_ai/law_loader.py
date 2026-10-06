"""《中华人民共和国劳动合同法》下载、解析与落盘。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

LAW_NAME = "中华人民共和国劳动合同法"
LAW_VERSION = "2012年12月28日修订，现行有效"
OFFICIAL_SOURCE_URL = (
    "https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/fl/"
    "202011/t20201102_394622.html"
)

_CN_NUMBER = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100}
_NUMBER_CHARS = "零〇一二三四五六七八九十百"
_ARTICLE_RE = re.compile(rf"^第([{_NUMBER_CHARS}]+)条\s*(.*)$")
_CHAPTER_RE = re.compile(rf"^第([{_NUMBER_CHARS}]+)章\s*(.*)$")
_SECTION_RE = re.compile(rf"^第([{_NUMBER_CHARS}]+)节\s*(.*)$")


@dataclass(frozen=True)
class LawArticle:
    """一条法律条文及其可用于 RAG 的来源元数据。"""

    article_number: int
    article_label: str
    chapter: str
    section: str | None
    content: str
    law_name: str = LAW_NAME
    version: str = LAW_VERSION
    source_url: str = OFFICIAL_SOURCE_URL
    issuing_authority: str = "全国人民代表大会常务委员会"
    source_publisher: str = "中华人民共和国人力资源和社会保障部"


def chinese_number_to_int(value: str) -> int:
    """把一至九十八等中文序号转换成整数。"""

    if not value or any(char not in _CN_NUMBER | _CN_UNIT for char in value):
        raise ValueError(f"无效中文数字：{value!r}")

    result = 0
    pending = 0
    for char in value:
        if char in _CN_NUMBER:
            pending = _CN_NUMBER[char]
        else:
            unit = _CN_UNIT[char]
            result += (pending or 1) * unit
            pending = 0
    return result + pending


def _normalized_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.replace("\u00a0", " ").replace("\u3000", " ")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def extract_law_text(html: str) -> str:
    """从人社部页面中提取法规正文。"""

    soup = BeautifulSoup(html, "html.parser")
    candidates = [
        soup.select_one(".TRS_Editor"),
        soup.select_one("#zoom"),
        soup.select_one(".article-content"),
    ]
    candidates = [node for node in candidates if node is not None]
    if not candidates:
        raise ValueError("页面中未找到法规正文区域")

    node = max(candidates, key=lambda item: len(item.get_text(" ", strip=True)))
    text = node.get_text("\n", strip=True)
    if "第一条" not in text or "第九十八条" not in text:
        raise ValueError("法规正文不完整：未同时找到第一条和第九十八条")
    return "\n".join(_normalized_lines(text))


def parse_articles(text: str) -> list[LawArticle]:
    """按章、节和条文解析法规正文。"""

    articles: list[LawArticle] = []
    chapter = ""
    section: str | None = None
    article_number: int | None = None
    article_label = ""
    paragraphs: list[str] = []

    def flush_article() -> None:
        nonlocal article_number, article_label, paragraphs
        if article_number is None:
            return
        content = "\n".join(paragraphs).strip()
        if not content:
            raise ValueError(f"{article_label}缺少正文")
        articles.append(
            LawArticle(
                article_number=article_number,
                article_label=article_label,
                chapter=chapter,
                section=section,
                content=content,
            )
        )
        article_number = None
        article_label = ""
        paragraphs = []

    for line in _normalized_lines(text):
        chapter_match = _CHAPTER_RE.match(line)
        if chapter_match:
            flush_article()
            chapter = f"第{chapter_match.group(1)}章 {chapter_match.group(2)}".strip()
            section = None
            continue

        section_match = _SECTION_RE.match(line)
        if section_match:
            flush_article()
            section = f"第{section_match.group(1)}节 {section_match.group(2)}".strip()
            continue

        article_match = _ARTICLE_RE.match(line)
        if article_match:
            flush_article()
            article_number = chinese_number_to_int(article_match.group(1))
            article_label = f"第{article_match.group(1)}条"
            inline_content = article_match.group(2).strip()
            if inline_content:
                paragraphs.append(inline_content)
            continue

        if article_number is not None:
            paragraphs.append(line)

    flush_article()

    numbers = [article.article_number for article in articles]
    if numbers != list(range(1, 99)):
        missing = sorted(set(range(1, 99)) - set(numbers))
        duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
        raise ValueError(
            f"条文序号校验失败：共{len(articles)}条，缺失={missing}，重复={duplicates}"
        )
    return articles


def _solve_mohrss_cookie_challenge(html: str) -> dict[str, str]:
    """解析人社部页面当前使用的轻量 Cookie 挑战。"""

    fields = {}
    for name in ("WTKkN", "bOYDu", "wyeCN"):
        match = re.search(rf"{name}:(\d+)", html)
        if not match:
            raise ValueError("无法解析人社部页面会话挑战")
        fields[name] = int(match.group(1))

    ssid_match = re.search(r't,\s*(\d+)\);continue;case"4"', html)
    if not ssid_match:
        raise ValueError("无法解析人社部页面 EO_Bot_Ssid")

    return {
        "__tst_status": f"{sum(fields.values())}#",
        "EO_Bot_Ssid": ssid_match.group(1),
    }


def download_official_html(timeout: int = 30) -> str:
    """下载人社部发布的《劳动合同法》页面，包括会话挑战处理。"""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            )
        }
    )
    response = session.get(OFFICIAL_SOURCE_URL, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    html = response.text

    if "TRS_Editor" not in html and "EO_Bot_Ssid" in html:
        cookies = _solve_mohrss_cookie_challenge(html)
        for name, value in cookies.items():
            session.cookies.set(name, value, domain="www.mohrss.gov.cn")
        response = session.get(OFFICIAL_SOURCE_URL, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        html = response.text

    if "TRS_Editor" not in html:
        raise ValueError("人社部页面未返回可解析的法规正文")
    return html


def save_law_dataset(
    html: str,
    raw_html_path: Path,
    clean_text_path: Path,
    json_path: Path,
) -> list[LawArticle]:
    """保存原始网页、清洗文本和结构化 JSON 数据。"""

    law_text = extract_law_text(html)
    articles = parse_articles(law_text)

    for path in (raw_html_path, clean_text_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    raw_html_path.write_text(html, encoding="utf-8")
    clean_text_path.write_text(law_text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps([asdict(article) for article in articles], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return articles


def article_summary(articles: Iterable[LawArticle]) -> dict[str, object]:
    """生成便于命令行展示的数据集摘要。"""

    article_list = list(articles)
    return {
        "law_name": LAW_NAME,
        "version": LAW_VERSION,
        "article_count": len(article_list),
        "chapters": list(dict.fromkeys(article.chapter for article in article_list)),
        "source_url": OFFICIAL_SOURCE_URL,
    }

