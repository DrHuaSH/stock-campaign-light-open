"""
配置管理模块 - OpenAI 兼容版本
"""
import os
import json
from pathlib import Path

# 目录配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DNA_DIR = BASE_DIR / "dna_history"
HISTORY_DIR = BASE_DIR / "analysis_history"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
DNA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# API密钥配置文件
CONFIG_FILE = BASE_DIR / "user_config.json"


def load_api_keys():
    """加载保存的API密钥"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "tushare_token": "",
        "openai_api_key": "",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-3.5-turbo"
    }


def save_api_keys(tushare_token: str, openai_api_key: str, openai_base_url: str, openai_model: str):
    """保存API密钥到本地"""
    config = {
        "tushare_token": tushare_token,
        "openai_api_key": openai_api_key,
        "openai_base_url": openai_base_url,
        "openai_model": openai_model
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return True


# HMM模型配置
HMM_CONFIG = {
    "n_states": 4,  # 吸筹、拉升、派发、观望
    "n_iter": 100,
    "min_history_days": 60,  # 最少需要60天历史数据
    "feature_window": 20,    # 特征计算窗口
    "default_days": 1000,    # 默认获取1000天数据（约4年，覆盖完整周期）
    "min_data_threshold": 0.5,  # 数据获取阈值：实际获取达到目标的50%即可
}

# 阶段名称映射
STAGE_NAMES = {
    0: "吸筹期",
    1: "拉升期", 
    2: "派发期",
    3: "观望期"
}

STAGE_NAMES_EN = {
    0: "accumulation",
    1: "markup",
    2: "distribution", 
    3: "observation"
}
