# 法规知识库素材

当前知识库仅包含《中华人民共和国劳动合同法》。

- 发布来源：中华人民共和国人力资源和社会保障部
- 官方页面：https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/fl/202011/t20201102_394622.html
- 法规版本：2007年6月29日通过，2012年12月28日修订，现行有效
- 条文范围：第一条至第九十八条

文件说明：

- `raw/labor_contract_law.html`：下载的官方原始网页
- `processed/labor_contract_law.txt`：清洗后的法规正文
- `processed/labor_contract_law.json`：按条文结构化的数据，保留章、节、条文号和来源元数据

重新获取并解析：

```powershell
.\.venv\Scripts\python.exe scripts\fetch_law.py
```

