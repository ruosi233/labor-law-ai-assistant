"""供 LangGraph Agent 调用的 LangChain 工具。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from langchain_core.tools import BaseTool, tool

from labor_law_ai.calculator import CompensationType, calculate_compensation
from labor_law_ai.rag import LaborLawRetriever


def create_rag_tool(retriever: LaborLawRetriever) -> BaseTool:
    @tool("retrieve_labor_contract_law")
    def retrieve_labor_contract_law(query: str) -> list[dict]:
        """检索《中华人民共和国劳动合同法》原文及条文来源。"""

        return [result.to_dict() for result in retriever.search(query)]

    return retrieve_labor_contract_law


@tool("calculate_labor_compensation")
def calculate_labor_compensation(
    start_date: str,
    end_date: str,
    average_monthly_salary: float,
    compensation_type: str,
    notice_monthly_salary: float | None = None,
    local_average_salary: float | None = None,
) -> dict:
    """计算劳动合同解除的 N、N+1 或 2N 金额，不负责判断类型是否适用。"""

    result = calculate_compensation(
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        average_monthly_salary=Decimal(str(average_monthly_salary)),
        compensation_type=CompensationType(compensation_type),
        notice_monthly_salary=(
            Decimal(str(notice_monthly_salary))
            if notice_monthly_salary is not None
            else None
        ),
        local_average_salary=(
            Decimal(str(local_average_salary))
            if local_average_salary is not None
            else None
        ),
    )
    return result.to_dict()

