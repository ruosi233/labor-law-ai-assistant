"""对本地知识库和 LangGraph Agent 做端到端冒烟测试。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from labor_law_ai.agent import LaborLawAgent
from labor_law_ai.rag import LaborLawRetriever


def main() -> None:
    retriever = LaborLawRetriever()
    if retriever.count() != 98:
        raise RuntimeError(f"期望 98 条索引记录，实际为 {retriever.count()} 条")

    exact = retriever.search("《劳动合同法》第八十七条如何规定？")
    if not exact or exact[0].metadata["article_number"] != 87 or exact[0].score != 1.0:
        raise RuntimeError("第八十七条精确召回失败")

    agent = LaborLawAgent(retriever=retriever, llm=None)
    cases = [
        "公司一直没有和我签书面劳动合同，法律有什么规定？",
        (
            "公司违法解除我，入职日期：2020-01-01；解除日期：2024-01-01；"
            "平均月工资：10000元；按2N计算。"
        ),
    ]
    for index, query in enumerate(cases, start=1):
        result = agent.invoke(query, thread_id=f"smoke-{index}")
        answer = str(result.get("answer", ""))
        if not answer:
            raise RuntimeError(f"第 {index} 个 Agent 用例没有生成回答")
        print(f"用例 {index} 路由：{result['action']}")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
