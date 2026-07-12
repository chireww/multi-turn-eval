# RAG 多轮对话评测

基于 DeepEval + 百炼 qwen-plus 的知识库问答多轮对话评测。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入百炼和 Dify Key
```

## 评测管线

```
conversation_queries.yaml（9个多轮场景）
  │
  ▼
python fetch_conversation.py    → 调Dify API（带conversation_id）→ conversation_data.json
  │
  ▼
python run_conversation_eval.py → 6个指标评测 → conversation_eval_results.json
```

## 评测指标（6个）

**逐轮RAG指标（4个）**
- Faithfulness — 每轮答案是否忠于召回文档
- AnswerRelevancy — 每轮答案是否切题
- HallucinationCheck — 是否有过度推断
- ComplianceCheck — 是否有医学违规词

**跨轮检查（2个）**
- 上下文记忆 — 后续轮次是否关联了前文信息
- 话题隔离 — 切换话题后是否被旧话题污染

## 多轮场景（9个）

| ID | 场景 | 轮数 |
|------|------|------|
| MC001 | 指代消解 | 3 |
| MC002 | 上下文延续 | 3 |
| MC003 | 话题切换 | 3 |
| MC004 | 用户纠错 | 2 |
| MC005 | 空召回后追问 | 2 |
| MC006 | 空召回后追问（备用） | 2 |
| MC007 | 多跳推理 | 3 |
| MC008 | 歧义澄清 | 2 |
| MC009 | 长对话记忆 | 5 |
