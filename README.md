# 劳动用工法律 AI 助手

基于 Python、LangGraph、LangChain、BGE 中文向量模型和 Chroma 构建。

## 当前能力

- 《中华人民共和国劳动合同法》98 条法规已结构化，每条记录包含法规名称、条文号和法条原文。
- RAG 检索支持相似度阈值过滤、精确条文匹配和法条来源引用。
- 赔偿计算器支持 N、N+1、2N，并输出年限、基数、封顶规则和计算过程。
- LangGraph Agent 支持四条路径：直接回答、检索法条、赔偿计算、缺失信息追问。
- 同一会话会保留案情，用户补充日期、工资等信息后可继续计算。
- Streamlit 提供聊天界面和 Agent 执行轨迹。

## Agent 流程

```text
用户问题
  -> 分析与路由
     -> 简单概念 ---------> 生成回答
     -> 法律问题 -> RAG --> 观察结果 -> 生成回答 -> 引用校验
     -> 赔偿问题 -> 信息检查
                    -> 信息不足 -> 主动追问
                    -> 信息完整 -> 计算工具 -> 生成回答 -> 校验
```

## 构建知识库

首次使用时，将中文向量模型下载到项目内：

```powershell
modelscope download AI-ModelScope/bge-base-zh-v1.5 `
  --local-dir .\models\bge-base-zh-v1.5
```

使用 GPU 构建或重建 Chroma 索引：

```powershell
python scripts\build_index.py --reset
```

重复执行但不带 `--reset` 时，只补充缺失记录，不会重复写入。可以运行端到端验证：

```powershell
python scripts\smoke_test.py
```

## 启动

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

项目没有配置大模型密钥时，会使用本地规则完成路由、RAG 和计算；配置兼容 OpenAI 接口的模型后，会启用大模型做结构化路由和回答生成。

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。

> 本项目用于法律信息检索和演示，不替代律师意见。
