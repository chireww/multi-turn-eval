# -*- coding: utf-8 -*-
"""统一配置：从 .env 文件加载，所有脚本共用。"""
import os
from dotenv import load_dotenv

# 加载 .env（优先找同目录，找不到就往上找）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"缺少环境变量: {key}，请检查 .env 文件")
    return val


# 百炼
BAILIAN_API_KEY = _require("BAILIAN_API_KEY")
BAILIAN_BASE_URL = _require("BAILIAN_BASE_URL")
JUDGE_MODEL = _require("JUDGE_MODEL")

# Dify
DIFY_API_KEY = _require("DIFY_API_KEY")
DIFY_BASE_URL = _require("DIFY_BASE_URL")
