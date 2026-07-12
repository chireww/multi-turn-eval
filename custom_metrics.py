# -*- coding: utf-8 -*-
"""
自定义 RAG 评测指标：Recall@K / Noise Robustness / Hallucination
=================================================================
不用 LLM 裁判，直接逻辑比对。和 DeepEval 四指标互补。
"""

import requests


# ================================================================
# 1. Recall@K：前 K 个召回文档覆盖了多少关键信息点
# ================================================================
def recall_at_k(contexts, info_points, k=3):
    """
    contexts:     召回文档列表，["文档1内容", "文档2内容", ...]
    info_points:  预期答案里的关键信息点，["2-8℃", "避光保存", ...]
    k:            取前 K 个文档
    返回:         0~1 之间的分数
    """
    if not contexts or not info_points:
        return None  # 空召回，指标无意义

    top_k_text = " ".join(contexts[:k])
    hits = sum(1 for p in info_points if p in top_k_text)
    return hits / len(info_points)


# ================================================================
# 2. Noise Robustness：塞噪音后答案是否被干扰
# ================================================================
# 预设的噪音文档片段（跟你的知识库完全无关的内容）
NOISE_SNIPPETS = [
    "公司年会在2025年12月25日于国际会议中心举行，请全体员工准时参加。",
    "今日菜谱：宫保鸡丁、鱼香肉丝、清炒时蔬、紫菜蛋花汤。",
    "请各部门于周五前提交季度工作报告，逾期将影响绩效考核。",
    "地铁3号线将于本周末进行检修，请乘客提前规划出行路线。",
]


def noise_robustness(dify_api_key, dify_base_url, query, clean_contexts, k=2):
    """
    query:           用户问题
    clean_contexts:  正常的召回文档列表
    k:               塞几条噪音
    返回:            (是否被干扰, 污染后的答案, 污染后的召回文档)
    """
    if not clean_contexts:
        return None  # 正常召回就为空，不用测

    # 注入噪音
    noisy_contexts = clean_contexts + NOISE_SNIPPETS[:k]

    # 重新调 Dify（但绕过正常检索，这里假设 Dify API 不支持手动指定 contexts）
    # 实际做法：用同样的问题重跑，拿到新答案，人工判断是否被干扰
    # 此处用代码标记为"需要人工对比"场景

    return {
        "query": query,
        "noise_injected": NOISE_SNIPPETS[:k],
        "status": "manual_check",  # 需要人工对比 clean_answer vs noisy_answer
        "note": "重新跑 Dify 获取 noisy_answer，人工对比核心事实是否一致",
    }


# ================================================================
# 3. Hallucination Check：过度推断检测
# ================================================================
HALLUCINATION_PATTERNS = {
    # 召回文档里的保守表述 → 答案里的绝对化表述 = 过度推断
    "可能": ["一定", "肯定", "必然", "毫无疑问"],
    "有助于": ["可以", "能够有效", "是有效的"],
    "部分患者": ["所有患者", "患者", "普遍"],
    "研究表明": ["已证实", "已经证明", "明确"],
    "发生率较低": ["安全", "几乎没有", "非常安全"],
}


def check_hallucination(actual_output, contexts):
    """
    检查答案里有没有'过度推断'——文档用了保守表述，答案却用了绝对化表述。

    actual_output:  系统生成的回答
    contexts:       召回文档列表
    返回:           [(触发词, 过度推断词, 上下文片段), ...]
    """
    if not contexts:
        return []

    all_context_text = " ".join(contexts)
    findings = []

    for cautious_word, exaggerated_words in HALLUCINATION_PATTERNS.items():
        if cautious_word in all_context_text:
            for exaggerated in exaggerated_words:
                if exaggerated in actual_output:
                    # 找到被过度推断的句子
                    for sentence in actual_output.split("。"):
                        if exaggerated in sentence:
                            findings.append({
                                "type": "过度推断",
                                "context_used": cautious_word,
                                "answer_used": exaggerated,
                                "sentence": sentence.strip(),
                            })

    return findings


# ================================================================
# 4. MRR（Mean Reciprocal Rank）
#    第一个相关文档排在第几位？1/rank 的均值
# ================================================================
def calc_mrr(contexts: list[str], info_points: list[str]) -> float | None:
    """
    按 Dify 返回的顺序（已按 score 降序），找第一个包含任意 info_point 的文档。
    返回 1/rank。如果都没命中，返回 0。
    """
    if not contexts or not info_points:
        return None
    for i, ctx in enumerate(contexts):
        for point in info_points:
            if point.lower() in ctx.lower():
                return 1.0 / (i + 1)  # rank = i+1（1-indexed）
    return 0.0  # 全部没命中


# ================================================================
# 5. Answer Correctness（模拟版）
#    真实版需要医学团队金标准 + LLM 裁判
#    模拟版：答案跟 ground_truth 的 info_points 覆盖比例
# ================================================================
def answer_correctness(answer: str, ground_truth: str, info_points: list[str]) -> dict:
    """模拟 Answer Correctness。真实版需 LLM 裁判 + 医学标准。"""
    if not ground_truth or not info_points:
        return {"score": None, "detail": "无 ground_truth，跳过（真实版需医学团队提供金标准）"}

    # 检查答案覆盖了 ground_truth 中多少 info_points
    hits = [p for p in info_points if p.lower() in answer.lower()]
    score = len(hits) / len(info_points)

    return {
        "score": round(score, 2),
        "hit": hits,
        "miss": [p for p in info_points if p.lower() not in answer.lower()],
        "detail": f"（模拟）答案覆盖了 ground_truth 中 {len(hits)}/{len(info_points)} 个信息点",
        "note": "真实场景应使用 LLM 裁判 + 医学团队金标准",
    }


# ================================================================
# 6. 医学合规词检测
# ================================================================
FORBIDDEN_WORDS = [
    "保证治愈", "100%有效", "绝对安全", "无任何副作用",
    "根治", "永不复发", "特效药", "神药", "偏方",
    "一定不会", "肯定能好", "保证治好",
]


def check_compliance(answer: str) -> list[str]:
    """检查答案里是否有违规表述。返回违规词列表。"""
    violations = []
    for word in FORBIDDEN_WORDS:
        if word in answer:
            violations.append(word)
    return violations


# ================================================================
# 运行全部自定义指标
# ================================================================
def run_custom_metrics(dify_data, info_points_map, ground_truth_map=None):
    """
    dify_data:         来自 dify_rag_data.json 的数据列表
    info_points_map:   {query: [关键信息点列表], ...}
    ground_truth_map:  {query: ground_truth, ...}  可选，给 Answer Correctness 用
    返回:              自定义指标结果列表
    """
    if ground_truth_map is None:
        ground_truth_map = {}
    custom_results = []

    for record in dify_data:
        query = record["question"]
        answer = record["answer"]
        contexts = record["contexts"]

        result = {
            "question": query,
            "custom_metrics": [],
        }

        # ---- Recall@K ----
        info_points = info_points_map.get(query, [])
        if info_points:
            r3 = recall_at_k(contexts, info_points, k=3)
            result["custom_metrics"].append({
                "name": "Recall@3",
                "score": round(r3, 2) if r3 is not None else None,
                "info_points": info_points,
                "detail": f"前3篇覆盖了 {int(r3 * len(info_points)) if r3 else 0}/{len(info_points)} 个信息点" if r3 is not None else "召回为空，跳过",
            })

            if len(contexts) >= 5:
                r5 = recall_at_k(contexts, info_points, k=5)
                result["custom_metrics"].append({
                    "name": "Recall@5",
                    "score": round(r5, 2),
                    "detail": f"前5篇覆盖了 {int(r5 * len(info_points))}/{len(info_points)} 个信息点",
                })

            # ---- MRR：第一个相关文档排第几位 ----
            mrr = calc_mrr(contexts, info_points)
            result["custom_metrics"].append({
                "name": "MRR",
                "score": round(mrr, 2) if mrr is not None else None,
                "detail": f"第一个相关文档在第 {int(1/mrr)} 位" if (mrr and mrr > 0) else ("首个相关文档未在前列" if mrr == 0 else "召回为空，跳过"),
            })

        # ---- Hallucination Check ----
        hallucinations = check_hallucination(answer, contexts)
        h_score = 1.0 if not hallucinations else max(0, 1.0 - len(hallucinations) * 0.3)
        result["custom_metrics"].append({
            "name": "HallucinationCheck",
            "score": round(h_score, 2),
            "findings": hallucinations,
            "detail": f"发现 {len(hallucinations)} 处过度推断" if hallucinations else "无过度推断",
        })

        # ---- Completeness：info_points 在答案中的覆盖率 ----
        if info_points:
            answer_text = answer.lower()
            hits = [p for p in info_points if p.lower() in answer_text]
            c_score = len(hits) / len(info_points)
            result["custom_metrics"].append({
                "name": "Completeness",
                "score": round(c_score, 2),
                "hit": hits,
                "miss": [p for p in info_points if p.lower() not in answer_text],
                "detail": f"答案覆盖了 {len(hits)}/{len(info_points)} 个关键信息点" + (f"，遗漏: {[p for p in info_points if p.lower() not in answer_text]}" if len(hits) < len(info_points) else "，全部覆盖"),
            })

        # ---- Answer Correctness（模拟版） ----
        gt = ground_truth_map.get(query, "")
        ac = answer_correctness(answer, gt, info_points)
        result["custom_metrics"].append({
            "name": "AnswerCorrectness",
            "score": ac["score"],
            **{k: v for k, v in ac.items() if k != "score"},
        })

        # ---- 医学合规词检测 ----
        violations = check_compliance(answer)
        result["custom_metrics"].append({
            "name": "ComplianceCheck",
            "score": 1.0 if not violations else 0.0,
            "violations": violations,
            "detail": f"发现 {len(violations)} 处违规词" if violations else "无违规词",
        })

        # ---- Noise Robustness（标记为需人工验证） ----
        result["custom_metrics"].append({
            "name": "NoiseRobustness",
            "score": None,
            "detail": "需人工验证：往召回结果塞 2 条无关文本后，重新跑 Dify，对比答案是否被干扰",
        })

        custom_results.append(result)

    return custom_results
