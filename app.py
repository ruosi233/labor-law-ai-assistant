"""劳动用工法律 AI 助手 Streamlit 界面。"""

from __future__ import annotations

import uuid

import streamlit as st

from labor_law_ai.agent import LaborLawAgent
from labor_law_ai.config import get_settings


st.set_page_config(
    page_title="劳动用工法律 AI 助手",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1050px; padding-top: 2rem;}
    .status-card {padding: .75rem 1rem; border: 1px solid #e5e7eb;
                  border-radius: .75rem; background: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent() -> LaborLawAgent:
    return LaborLawAgent(settings=get_settings())


def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.last_trace = []
    st.session_state.last_sources = []


if "thread_id" not in st.session_state:
    reset_conversation()

settings = get_settings()
agent = get_agent()

with st.sidebar:
    st.header("运行状态")
    st.markdown(
        f"""
        <div class="status-card">
        <b>向量模型</b><br>{settings.embedding_model}<br><br>
        <b>计算设备</b><br>{settings.embedding_device.upper()}<br><br>
        <b>大模型</b><br>{settings.llm_model if settings.llm_api_key else '未配置，使用本地降级模式'}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("新建会话", use_container_width=True, on_click=reset_conversation)
    st.divider()
    st.caption("知识库范围：仅《中华人民共和国劳动合同法》")
    st.caption("回答仅供信息参考，不构成正式法律意见。")

st.title("⚖️ 劳动用工法律 AI 助手")
st.caption("LangGraph 多步骤 Agent · 法条 RAG · N / N+1 / 2N 计算")

with st.expander("可以这样问", expanded=not st.session_state.chat_history):
    st.markdown(
        """
        - 什么是无固定期限劳动合同？
        - 公司一直没和我签劳动合同，法律怎么规定？
        - 《劳动合同法》第八十七条是什么？
        - 公司辞退我能赔多少钱？
        """
    )

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if prompt := st.chat_input("请描述你的劳动用工问题……"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在分析并选择工具……"):
            try:
                state = agent.invoke(prompt, st.session_state.thread_id)
                answer = state.get("answer", "暂时无法生成回答。")
                st.markdown(answer)
                st.session_state.last_trace = state.get("trace", [])
                st.session_state.last_sources = state.get("retrieved_articles", [])
            except Exception as exc:
                answer = f"处理失败：{exc}"
                st.error(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

if st.session_state.last_trace:
    with st.expander("本轮 Agent 执行步骤"):
        for index, step in enumerate(st.session_state.last_trace, start=1):
            st.write(f"{index}. {step}")

