# -*- coding: utf-8 -*-
"""
多轮对话：从 convesation_queries.yaml 加载场景 → 调 Dify API（带 conversation_id）→ 保存 JSON。
"""
import json
import time
import requests
import yaml

from config import DIFY_API_KEY, DIFY_BASE_URL
YAML_FILE = "conversation_queries.yaml"
OUTPUT_FILE = "conversation_data.json"
NO_PROXY = {"http": None, "https": None}


def load_scenarios(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"加载 {len(data['scenarios'])} 个多轮对话场景")
    return data["scenarios"]


def send_message(query: str, conversation_id: str = "", user: str = "eval-user") -> dict:
    """发送一条消息，返回 Dify 的完整响应。"""
    payload = {
        "query": query,
        "user": user,
        "response_mode": "blocking",
        "inputs": {},
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id

    resp = requests.post(
        f"{DIFY_BASE_URL}/chat-messages",
        headers={
            "Authorization": f"Bearer {DIFY_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=320,
        proxies=NO_PROXY,
    )
    resp.raise_for_status()
    return resp.json()


def run_scenario(scenario: dict) -> dict:
    """运行一个多轮对话场景，返回完整对话记录。"""
    print(f"\n{'='*50}")
    print(f"场景: {scenario['id']} — {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print(f"{'='*50}")

    conversation_id = ""
    turns_data = []

    for i, turn in enumerate(scenario["turns"]):
        user_msg = turn["user"]
        print(f"\n  第{i+1}轮 用户: {user_msg}")

        data = send_message(user_msg, conversation_id)
        conversation_id = data.get("conversation_id", "")       # Dify 返回的，后续轮次带上

        answer = data.get("answer", "")
        retriever_resources = data.get("metadata", {}).get("retriever_resources", [])
        contexts = [r.get("content", "") for r in retriever_resources]

        turns_data.append({
            "turn": i + 1,
            "user_message": user_msg,
            "answer": answer,
            "contexts": contexts,
            "conversation_id": conversation_id,
        })

        print(f"    答案: {answer[:100]}...")
        print(f"    召回片段: {len(contexts)} 条")
        time.sleep(0.3)

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "description": scenario["description"],
        "turns": turns_data,
    }


def main():
    scenarios = load_scenarios(YAML_FILE)

    all_results = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        all_results.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"完成 {len(all_results)} 个场景的多轮对话")
    print(f"数据已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
