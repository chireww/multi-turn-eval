# -*- coding: utf-8 -*-
"""
多轮对话评测：加载 conversation_data.json → 逐轮跑 RAG 指标 + 跨轮检查。
"""
import json
import requests
import asyncio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===== 配置 =====
from config import BAILIAN_API_KEY, BAILIAN_BASE_URL, JUDGE_MODEL
DATA_FILE = "conversation_data.json"

from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM


# ===== 带重试的 Session（和 run_eval.py 一样）=====
def build_retry_session():
    session = requests.Session()
    session.trust_env = False
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 10054],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


REQUEST_SESSION = build_retry_session()


# ===== 裁判模型（带重试）=====
class BailianModel(DeepEvalBaseLLM):
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def load_model(self):
        return self

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt):
        resp = REQUEST_SESSION.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model_name,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"API {resp.status_code}: {resp.text}")

    async def a_generate(self, prompt_or_messages, **kwargs):
        kwargs.pop("schema", None)
        kwargs.pop("response_format", None)
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages

        def _sync():
            resp = REQUEST_SESSION.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model_name, "messages": messages, "temperature": 0},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"], 0.0
            raise RuntimeError(f"API {resp.status_code}: {resp.text}")

        return await asyncio.to_thread(_sync)


# ===== 多轮特有检查 =====
def check_context_memory(turns_data):
    """
    跨轮记忆检查：第 N 轮的答案是否关联了第 N-1 轮的内容。
    简单规则：第 N 轮答案 + 第 N-1 轮召回文档 → Faithfulness。
    如果第 N 轮能基于上一轮的召回文档回答，说明有上下文记忆。
    """
    results = []
    for i in range(1, len(turns_data)):
        prev = turns_data[i - 1]
        curr = turns_data[i]
        # 简单判断：当前轮答案里是否包含上一轮的关键词
        prev_keywords = set(prev["answer"]) & set(curr["answer"])
        overlap = len(prev_keywords) / max(len(set(prev["answer"])), 1)
        results.append({
            "from_turn": i,
            "to_turn": i + 1,
            "prev_answer_overlap": round(overlap, 2),
            "note": "关键词重叠率（仅参考，需人工确认语义关联）" if overlap > 0.01 else "无重叠",
        })
    return results


def check_topic_isolation(turns_data):
    """
    话题隔离检查：第 N 轮的答案是否被前一轮话题污染。
    规则：如果第 N 轮跟前一轮话题完全不同，答案里不应出现前一轮的关键词。
    """
    results = []
    # 预定义的话题关键词（从 conversation_queries.yaml 的场景设计意图来）
    topic_keywords = {
        "胃癌": ["胃癌", "胃窦", "肿瘤", "手术", "化疗", "免疫", "辅助治疗"],
        "实验室": ["实验室", "刷卡", "门禁", "行政部"],
        "药品": ["药品", "不良反应", "副作用", "剂量", "用法", "禁忌"],
        "天气": ["天气", "温度", "下雨", "晴"],
        "医保": ["医保", "报销", "临床实验", "费用"],
    }

    for i in range(len(turns_data)):
        curr = turns_data[i]
        answer_lower = curr["answer"]
        found_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in answer_lower for kw in keywords):
                found_topics.append(topic)
        results.append({
            "turn": i + 1,
            "detected_topics": found_topics,
            "note": "人工确认这些话题是否应该出现在此轮回答中",
        })
    return results


# ===== 引入自定义指标 =====
from custom_metrics import check_hallucination, check_compliance


# ===== 主流程 =====
def main():
    model = BailianModel(JUDGE_MODEL, BAILIAN_API_KEY, BAILIAN_BASE_URL)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"加载 {len(all_data)} 个多轮对话场景\n")

    all_eval_results = []

    for scenario_data in all_data:
        sid = scenario_data["scenario_id"]
        print(f"{'='*60}")
        print(f"场景: {sid} — {scenario_data['scenario_name']}")
        print(f"{'='*60}")

        scenario_result = {
            "scenario_id": sid,
            "scenario_name": scenario_data["scenario_name"],
            "turns": [],
            "cross_turn_checks": {},
        }

        # ---- 第一步：逐轮 RAG 评测 ----
        for turn in scenario_data["turns"]:
            print(f"\n  第{turn['turn']}轮: {turn['user_message'][:50]}...")

            turn_metrics = []
            if turn["contexts"]:  # 有召回才测
                case = LLMTestCase(
                    input=turn["user_message"],
                    actual_output=turn["answer"],
                    retrieval_context=turn["contexts"],
                )

                # DeepEval 指标：Precision/Recall 需要 expected_output，多轮没有就跳过
                metrics_to_run = [FaithfulnessMetric, AnswerRelevancyMetric]
                if case.expected_output:
                    metrics_to_run = [FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric]

                for metric_cls in metrics_to_run:
                    m = metric_cls(model=model, threshold=0.7)
                    try:
                        m.measure(case)
                        turn_metrics.append({
                            "metric": metric_cls.__name__,
                            "score": round(m.score, 2),
                            "passed": m.success,
                            "reason": m.reason,
                        })
                        print(f"    {metric_cls.__name__}: {m.score:.2f} {'✅' if m.success else '❌'}")
                    except Exception as e:
                        print(f"    {metric_cls.__name__}: 失败 - {str(e)[:60]}")

                # 自定义指标：HallucinationCheck
                hallucinations = check_hallucination(turn["answer"], turn["contexts"])
                h_score = 1.0 if not hallucinations else max(0, 1.0 - len(hallucinations) * 0.3)
                turn_metrics.append({
                    "metric": "HallucinationCheck",
                    "score": round(h_score, 2),
                    "findings": hallucinations,
                })
                print(f"    HallucinationCheck: {h_score:.2f} ({'无过度推断' if not hallucinations else f'{len(hallucinations)}处'})")

                # ComplianceCheck
                violations = check_compliance(turn["answer"])
                turn_metrics.append({
                    "metric": "ComplianceCheck",
                    "score": 1.0 if not violations else 0.0,
                    "violations": violations,
                })
                print(f"    ComplianceCheck: {'✅' if not violations else '❌ 违规词:' + ','.join(violations)}")
            else:
                print(f"    召回为空，跳过逐轮评测")
                print(f"    召回为空，跳过逐轮评测")

            scenario_result["turns"].append({
                "turn": turn["turn"],
                "user_message": turn["user_message"],
                "answer_preview": turn["answer"][:120],
                "single_turn_metrics": turn_metrics,
            })

        # ---- 第二步：多轮特有检查 ----
        print(f"\n  --- 跨轮检查 ---")
        memory = check_context_memory(scenario_data["turns"])
        topic = check_topic_isolation(scenario_data["turns"])
        scenario_result["cross_turn_checks"] = {
            "context_memory": memory,
            "topic_isolation": topic,
        }
        for m in memory:
            print(f"    第{m['from_turn']}→{m['to_turn']}轮: {m['note']}")
        for t in topic:
            print(f"    第{t['turn']}轮话题: {t['detected_topics'] or '无明确话题'}")

        all_eval_results.append(scenario_result)

    # ---- 保存 ----
    output = "conversation_eval_results.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"📁 多轮评测结果已保存到 {output}")


if __name__ == "__main__":
    main()
