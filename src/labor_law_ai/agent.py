"""LangGraph 多步骤劳动用工法律 Agent。"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from labor_law_ai.calculator import (
    CompensationType,
    MissingCalculationFields,
    calculate_compensation,
)
from labor_law_ai.config import Settings, get_settings
from labor_law_ai.law_loader import OFFICIAL_SOURCE_URL
from labor_law_ai.rag import LaborLawRetriever


Action = Literal["direct", "rag", "calculate", "clarify"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    action: Action
    route_reason: str
    case_facts: dict[str, Any]
    missing_fields: list[str]
    retrieved_articles: list[dict[str, Any]]
    calculation_result: dict[str, Any] | None
    observation: str
    answer: str
    citations_valid: bool
    trace: list[str]


class RouteDecision(BaseModel):
    action: Action = Field(description="direct、rag、calculate 或 clarify")
    reason: str = Field(description="一句话说明所选处理路线，不输出思维过程")


class ExtractedCaseFacts(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    average_monthly_salary: float | None = None
    notice_monthly_salary: float | None = None
    local_average_salary: float | None = None
    termination_reason: str | None = None
    termination_actor: Literal["employer", "employee"] | None = None
    notice_given: bool | None = None
    compensation_type: Literal["N", "N+1", "2N"] | None = None


_CALCULATION_KEYWORDS = (
    "赔多少", "补偿多少", "赔偿多少", "多少钱", "计算补偿", "计算赔偿",
    "经济补偿金", "赔偿金", "n+1", "2n",
)
_RAG_KEYWORDS = (
    "法律依据", "法条", "第几条", "规定", "是否合法", "合法吗", "怎么办",
    "能不能", "可以吗", "应当", "是否可以",
)
_DIRECT_PREFIXES = ("什么是", "是什么意思", "概念", "区别", "如何理解")


def classify_intent(query: str) -> Action:
    """无模型时的可解释路由规则。"""

    normalized = query.strip().lower()
    if not normalized:
        return "clarify"
    if any(keyword in normalized for keyword in _CALCULATION_KEYWORDS):
        return "calculate"
    if normalized.startswith(_DIRECT_PREFIXES):
        return "direct"
    if any(keyword in normalized for keyword in _RAG_KEYWORDS) or re.search(
        r"第\s*[零〇一二三四五六七八九十百\d]+\s*条", normalized
    ):
        return "rag"
    return "rag"


def create_chat_model(settings: Settings) -> BaseChatModel | None:
    """仅在配置密钥时创建模型；否则使用本地确定性降级流程。"""

    if not settings.llm_api_key.strip():
        return None
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": 0,
    }
    if settings.llm_base_url.strip():
        kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


def _last_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _trace(state: AgentState, step: str) -> list[str]:
    return [*state.get("trace", []), step]


def _normalize_date(raw: str) -> str:
    parts = re.findall(r"\d+", raw)
    if len(parts) != 3:
        raise ValueError(f"无法识别日期：{raw}")
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def _amount_from_match(match: re.Match[str]) -> float:
    value = float(match.group(1))
    unit = match.group(2) or "元"
    return value * 10000 if unit == "万" else value


def extract_case_facts_by_rules(text: str) -> dict[str, Any]:
    """无大模型时从常见中文表达中提取计算字段。"""

    facts: dict[str, Any] = {}
    date_pattern = r"(\d{4}[年./-]\d{1,2}[月./-]\d{1,2}日?)"
    patterns = {
        "start_date": rf"(?:入职日期|入职时间|开始工作)\s*[:：]?\s*{date_pattern}",
        "end_date": rf"(?:解除日期|终止日期|离职日期|辞退日期)\s*[:：]?\s*{date_pattern}",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts[field] = _normalize_date(match.group(1))

    amount_patterns = {
        "average_monthly_salary": (
            r"(?:前十二个月平均工资|解除前十二个月平均工资|平均月工资|月平均工资)"
            r"\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(万|元)?"
        ),
        "notice_monthly_salary": (
            r"(?:解除前上一个月工资|上月工资|代通知金工资)"
            r"\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(万|元)?"
        ),
        "local_average_salary": (
            r"(?:当地上年度职工月平均工资|当地社平工资|社平工资)"
            r"\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(万|元)?"
        ),
    }
    for field, pattern in amount_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts[field] = _amount_from_match(match)

    type_match = re.search(r"(?<![a-z0-9])(N\+1|2N|N)(?![a-z0-9])", text, re.IGNORECASE)
    if type_match:
        facts["compensation_type"] = type_match.group(1).upper()

    if re.search(r"(?:没有|未|没)\s*(?:提前)?\s*30\s*天", text):
        facts["notice_given"] = False
    elif re.search(r"(?:已经|有|已)\s*(?:提前)?\s*30\s*天", text):
        facts["notice_given"] = True

    if re.search(r"(?:公司|单位|用人单位).{0,10}(?:解除|辞退|终止)", text):
        facts["termination_actor"] = "employer"
    elif re.search(r"(?:我|劳动者).{0,6}(?:辞职|主动离职|解除)", text):
        facts["termination_actor"] = "employee"

    explicit_reason = re.search(r"(?:解除原因|辞退原因|终止原因)\s*[:：]\s*([^；。\n]+)", text)
    if explicit_reason:
        facts["termination_reason"] = explicit_reason.group(1).strip()
    else:
        reason_keywords = (
            "违法解除", "违法辞退", "无故辞退", "不能胜任", "医疗期满",
            "客观情况发生重大变化", "合同到期", "协商一致", "经济性裁员",
            "拖欠工资", "未缴社保", "主动辞职",
        )
        found = next((keyword for keyword in reason_keywords if keyword in text), None)
        if found:
            facts["termination_reason"] = found

    return facts


def _infer_compensation_type(facts: dict[str, Any]) -> str | None:
    explicit = facts.get("compensation_type")
    if explicit:
        return str(explicit)
    reason = str(facts.get("termination_reason", ""))
    actor = facts.get("termination_actor")
    if actor == "employer" and any(word in reason for word in ("违法", "无故")):
        return "2N"
    if any(word in reason for word in ("不能胜任", "医疗期满", "客观情况")):
        if facts.get("notice_given") is True:
            return "N"
        if facts.get("notice_given") is False:
            return "N+1"
        return None
    if any(word in reason for word in ("合同到期", "协商一致", "经济性裁员")):
        return "N"
    return None


def _missing_calculation_fields(facts: dict[str, Any]) -> list[str]:
    required = [
        "start_date",
        "end_date",
        "average_monthly_salary",
        "termination_actor",
        "termination_reason",
    ]
    missing = [field for field in required if facts.get(field) in (None, "")]
    inferred = _infer_compensation_type(facts)
    if inferred:
        facts["compensation_type"] = inferred
    elif "termination_reason" not in missing:
        missing.append("compensation_type")
    if facts.get("compensation_type") == "N+1" and not facts.get("notice_monthly_salary"):
        missing.append("notice_monthly_salary")
    return missing


_FIELD_QUESTIONS = {
    "start_date": "入职日期",
    "end_date": "解除或终止日期",
    "average_monthly_salary": "解除前十二个月平均工资",
    "termination_actor": "由公司解除还是劳动者主动离职",
    "termination_reason": "公司给出的具体解除或终止原因",
    "compensation_type": "希望按 N、N+1 或 2N 计算；若不确定，请补充更具体的解除经过",
    "notice_monthly_salary": "解除前上一个月工资（用于代通知金）",
}

_FIELD_EXAMPLES = {
    "start_date": "入职日期：2023-09-20",
    "end_date": "解除或终止日期：2025-09-20",
    "average_monthly_salary": "解除前十二个月平均工资：8000元",
    "termination_actor": "解除方：公司解除",
    "termination_reason": "解除原因：公司无故解除",
    "compensation_type": "计算方式：2N（不确定可写“不清楚”）",
    "notice_monthly_salary": "解除前上一个月工资：8000元",
}


class LaborLawAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: LaborLawRetriever | None = None,
        llm: BaseChatModel | None = None,
        checkpointer: MemorySaver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or LaborLawRetriever(settings=self.settings)
        self.llm = llm if llm is not None else create_chat_model(self.settings)
        self.graph = self._build_graph(checkpointer or MemorySaver())

    def _build_graph(self, checkpointer: MemorySaver):
        builder = StateGraph(AgentState)
        builder.add_node("analyze", self._analyze)
        builder.add_node("retrieve_law", self._retrieve_law)
        builder.add_node("prepare_calculation", self._prepare_calculation)
        builder.add_node("calculate", self._calculate)
        builder.add_node("observe", self._observe)
        builder.add_node("respond", self._respond)
        builder.add_node("verify_citations", self._verify_citations)
        builder.add_node("ask_for_details", self._ask_for_details)

        builder.add_edge(START, "analyze")
        builder.add_conditional_edges(
            "analyze",
            lambda state: state["action"],
            {
                "direct": "respond",
                "rag": "retrieve_law",
                "calculate": "prepare_calculation",
                "clarify": "ask_for_details",
            },
        )
        builder.add_edge("retrieve_law", "observe")
        builder.add_conditional_edges(
            "prepare_calculation",
            lambda state: "missing" if state.get("missing_fields") else "ready",
            {"missing": "ask_for_details", "ready": "calculate"},
        )
        builder.add_edge("calculate", "observe")
        builder.add_edge("observe", "respond")
        builder.add_edge("respond", "verify_citations")
        builder.add_edge("verify_citations", END)
        builder.add_edge("ask_for_details", END)
        return builder.compile(checkpointer=checkpointer)

    def _analyze(self, state: AgentState) -> dict[str, Any]:
        query = _last_human_text(state.get("messages", []))
        if state.get("missing_fields") and state.get("action") == "calculate":
            action: Action = "calculate"
            reason = "继续补全上一轮计算所需案情"
        elif self.llm is not None:
            try:
                router = self.llm.with_structured_output(RouteDecision)
                decision = router.invoke(
                    [
                        SystemMessage(
                            content=(
                                "你是劳动法律助手的路由器。简单概念选direct；需要法条证据选rag；"
                                "涉及金额、N、N+1、2N选calculate；问题为空或无法判断选clarify。"
                                "reason只写一句可展示的路线说明，不写思维过程。"
                            )
                        ),
                        HumanMessage(content=query),
                    ]
                )
                action, reason = decision.action, decision.reason
            except Exception:
                action, reason = classify_intent(query), "使用本地规则完成路线选择"
        else:
            action, reason = classify_intent(query), "使用本地规则完成路线选择"
        return {
            "user_query": query,
            "action": action,
            "route_reason": reason,
            "retrieved_articles": [],
            "calculation_result": None,
            "observation": "",
            "answer": "",
            "citations_valid": False,
            "trace": _trace(state, f"判断动作：{action}"),
        }

    def _retrieve_law(self, state: AgentState) -> dict[str, Any]:
        try:
            results = [item.to_dict() for item in self.retriever.search(state["user_query"])]
            observation = f"检索到{len(results)}条达到阈值的法条片段"
        except Exception as exc:
            results = []
            observation = f"法规检索失败：{exc}"
        return {
            "retrieved_articles": results,
            "observation": observation,
            "trace": _trace(state, "执行工具：劳动合同法 RAG 检索"),
        }

    def _extract_case_facts(self, state: AgentState) -> dict[str, Any]:
        existing = dict(state.get("case_facts", {}))
        query = state["user_query"]
        extracted: dict[str, Any]
        if self.llm is not None:
            try:
                extractor = self.llm.with_structured_output(ExtractedCaseFacts)
                result = extractor.invoke(
                    [
                        SystemMessage(
                            content=(
                                "从用户劳动解除案情中提取明确出现的字段。不得猜测日期、工资、"
                                "解除原因或赔偿类型。日期统一为YYYY-MM-DD。"
                            )
                        ),
                        HumanMessage(
                            content=f"已有事实：{existing}\n用户最新补充：{query}"
                        ),
                    ]
                )
                extracted = result.model_dump(exclude_none=True)
            except Exception:
                extracted = extract_case_facts_by_rules(query)
        else:
            extracted = extract_case_facts_by_rules(query)
        existing.update(extracted)
        return existing

    def _prepare_calculation(self, state: AgentState) -> dict[str, Any]:
        facts = self._extract_case_facts(state)
        missing = _missing_calculation_fields(facts)
        return {
            "case_facts": facts,
            "missing_fields": missing,
            "trace": _trace(state, "检查计算字段完整性"),
        }

    def _calculate(self, state: AgentState) -> dict[str, Any]:
        facts = state["case_facts"]
        try:
            result = calculate_compensation(
                start_date=date.fromisoformat(facts["start_date"]),
                end_date=date.fromisoformat(facts["end_date"]),
                average_monthly_salary=Decimal(str(facts["average_monthly_salary"])),
                compensation_type=CompensationType(facts["compensation_type"]),
                notice_monthly_salary=facts.get("notice_monthly_salary"),
                local_average_salary=facts.get("local_average_salary"),
            )
            calculation = result.to_dict()
            observation = f"完成{result.compensation_type}计算，总额{result.total}元"
            missing: list[str] = []
        except MissingCalculationFields as exc:
            calculation = None
            observation = str(exc)
            missing = exc.fields
        except Exception as exc:
            calculation = None
            observation = f"计算失败：{exc}"
            missing = []
        return {
            "calculation_result": calculation,
            "observation": observation,
            "missing_fields": missing,
            "trace": _trace(state, "执行工具：N / N+1 / 2N 补偿计算"),
        }

    def _observe(self, state: AgentState) -> dict[str, Any]:
        return {"trace": _trace(state, f"观察结果：{state.get('observation', '')}")}

    def _direct_fallback(self, query: str) -> str:
        concepts = {
            "无固定期限劳动合同": "无固定期限劳动合同，是用人单位与劳动者约定无确定终止时间的劳动合同。它不等于不能解除，符合法定条件时仍可解除或终止。",
            "经济补偿": "经济补偿通常按劳动者在本单位的工作年限和解除或终止前十二个月平均工资计算，常被简称为 N。是否应支付取决于解除或终止原因。",
            "赔偿金": "违法解除或违法终止劳动合同时，赔偿金通常按经济补偿标准的二倍计算，常被简称为 2N。",
            "试用期": "试用期是劳动合同期限内双方相互考察的期间，期限、工资和解除条件均受到法律限制。",
        }
        for name, answer in concepts.items():
            if name in query:
                return answer
        return "这是一个劳动法律概念问题。请再写明你想了解的具体概念，我会用简短语言说明。"

    def _respond(self, state: AgentState) -> dict[str, Any]:
        action = state["action"]
        if action == "direct":
            if self.llm is not None:
                try:
                    response = self.llm.invoke(
                        [
                            SystemMessage(
                                content="简洁解释劳动法律概念，不虚构具体法条或案件结论。"
                            ),
                            HumanMessage(content=state["user_query"]),
                        ]
                    )
                    answer = str(response.content)
                except Exception:
                    answer = self._direct_fallback(state["user_query"])
            else:
                answer = self._direct_fallback(state["user_query"])
        elif action == "rag":
            retrieved = state.get("retrieved_articles", [])
            if not retrieved:
                answer = (
                    "当前《劳动合同法》知识库没有检索到达到相关度阈值的可靠法条，"
                    "或者本地索引尚未建立。我不会在缺少法条依据时猜测答案。"
                )
            elif self.llm is not None:
                context = "\n\n".join(item["page_content"] for item in retrieved)
                try:
                    response = self.llm.invoke(
                        [
                            SystemMessage(
                                content=(
                                    "你是中国劳动用工法律助手。只能依据提供的《劳动合同法》原文回答；"
                                    "先给结论，再说明适用条件和不确定点。不得引用上下文之外的条文。"
                                )
                            ),
                            HumanMessage(
                                content=f"用户问题：{state['user_query']}\n\n法条原文：\n{context}"
                            ),
                        ]
                    )
                    answer = str(response.content)
                except Exception:
                    answer = self._rag_fallback(retrieved)
            else:
                answer = self._rag_fallback(retrieved)
        else:
            result = state.get("calculation_result")
            if not result:
                answer = f"暂时无法完成计算：{state.get('observation', '信息不足')}"
            else:
                warnings = "\n".join(f"- {item}" for item in result["warnings"])
                answer = (
                    f"按当前信息，暂估为 **{result['compensation_type']}**。\n\n"
                    f"- 工作期间：{result['service_period']}\n"
                    f"- 补偿年限 N：{result['n_months']}个月\n"
                    f"- 计算基数：{result['salary_base']}元\n"
                    f"- 计算过程：`{result['formula']}`\n"
                    f"- 暂估总额：**{result['total']}元**\n\n"
                    f"注意事项：\n{warnings}"
                )
        return {
            "answer": answer,
            "trace": _trace(state, "生成回答"),
        }

    @staticmethod
    def _rag_fallback(retrieved: list[dict[str, Any]]) -> str:
        lines = ["根据当前检索到的《劳动合同法》原文："]
        for item in retrieved[:3]:
            label = item["metadata"]["article_label"]
            content = item["page_content"].split("\n", 2)[-1]
            lines.append(f"\n**{label}**：{content}")
        return "\n".join(lines)

    def _verify_citations(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        source_lines: list[str] = []
        seen: set[str] = set()
        if state["action"] == "rag":
            for item in state.get("retrieved_articles", []):
                metadata = item["metadata"]
                label = metadata["article_label"]
                if label in seen:
                    continue
                seen.add(label)
                source_lines.append(
                    f"- [《{metadata['law_name']}》{label}]({metadata['source_url']})"
                    f"（相关度 {float(item['score']):.0%}）"
                )
        elif state["action"] == "calculate" and state.get("calculation_result"):
            for basis in state["calculation_result"]["legal_basis"]:
                if basis in seen:
                    continue
                seen.add(basis)
                source_lines.append(f"- [{basis}]({OFFICIAL_SOURCE_URL})")

        if source_lines:
            answer = f"{answer}\n\n**法律依据**\n\n" + "\n".join(source_lines)
        valid = state["action"] == "direct" or bool(source_lines)
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "citations_valid": valid,
            "trace": _trace(state, "复核法条引用"),
        }

    def _ask_for_details(self, state: AgentState) -> dict[str, Any]:
        missing = state.get("missing_fields", [])
        if missing:
            questions = [f"- {_FIELD_QUESTIONS.get(field, field)}" for field in missing]
            examples = [
                _FIELD_EXAMPLES[field]
                for field in missing
                if field in _FIELD_EXAMPLES
            ]
            example_block = "；\n".join(examples)
            answer = (
                "为了避免误算，请补充以下信息：\n\n"
                + "\n".join(questions)
                + "\n\n**请按下面格式回复**（复制后替换成你的实际情况；不知道的项目写“不清楚”）：\n\n"
                + f"```text\n{example_block}\n```"
            )
        else:
            answer = "请具体说明你想咨询的劳动用工问题。"
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "trace": _trace(state, "主动追问缺失信息"),
        }

    def invoke(self, message: str, thread_id: str) -> AgentState:
        return self.graph.invoke(
            {"messages": [HumanMessage(content=message)], "trace": []},
            config={"configurable": {"thread_id": thread_id}},
        )
