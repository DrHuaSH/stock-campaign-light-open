# 📈 Stock Campaign Light Open

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/DrHuaSH/stock-campaign-light-open/releases)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.54+-red.svg)](https://streamlit.io/)

轻量版股票战役分析系统 - **OpenAI 兼容版本 v2.0**

基于 HMM 模型和**任何 OpenAI 兼容 LLM API** 的智能股票阶段识别工具。

> 🎉 **v2.0 更新**: 增强安装指南、修复 HMM 模型加载问题、优化启动性能、新增完整故障排查

## 🌟 功能特点

- **🔍 单股深度分析**: 生成股票DNA，精确判断当前阶段
- **🔥 增强三维度分析**: 技术面 + 市场面 + 信息面综合判断
- **📈 批量分析**: 同时分析多只股票，快速筛选
- **🧬 股票DNA**: 保存每只股票的HMM模型特征
- **🤖 OpenAI 兼容 LLM**: 支持 OpenAI、Kimi、DeepSeek、Ollama、Azure、Groq、SiliconFlow 等**任何 OpenAI 兼容 API**
- **📰 新闻情绪分析**: 自动搜索和分析相关新闻（Tavily API）
- **📊 市场面分析**: 对比大盘/板块走势、相对强度
- **📜 历史记录**: 本地保存所有分析记录
- **🎯 预测验证**: 验证历史预测准确性，持续优化模型
- **🔄 双数据源**: Tushare Pro + AKShare备份

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

首次运行时需要配置：

- **Tushare Pro Token**: 在 [Tushare官网](https://tushare.pro/register) 注册获取
- **LLM API 配置**: 支持**任何 OpenAI 兼容**的服务商
- **Tavily API Key** (可选): 用于新闻搜索，免费版每月1000次调用
  - 在 [Tavily官网](https://tavily.com/) 注册获取
  - 在侧边栏"📰 Tavily API 配置"中输入

| 服务商 | Base URL | 模型示例 |
|--------|----------|----------|
| **OpenAI** | `https://api.openai.com/v1` | `gpt-3.5-turbo`, `gpt-4`, `gpt-4o` |
| **Kimi (Moonshot)** | `https://api.moonshot.cn/v1` | `moonshot-v1-8k`, `moonshot-v1-32k` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-coder` |
| **SiliconFlow** | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| **Ollama (本地)** | `http://localhost:11434/v1` | `llama2`, `mistral`, `qwen` |
| **Azure OpenAI** | `https://your-resource.openai.azure.com/openai/deployments/your-deployment` | `gpt-35-turbo` |
| **Together AI** | `https://api.together.xyz/v1` | `meta-llama/Llama-3.1-8B-Instruct-Turbo` |
| **Fireworks** | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p1-8b-instruct` |
| **LocalAI** | `http://localhost:8080/v1` | 任何本地模型 |
| **vLLM** | `http://localhost:8000/v1` | 任何本地模型 |

> 💡 **提示**: 只要 API 遵循 OpenAI 的接口格式（`/chat/completions`），都可以直接使用！

### 3. 启动程序

```bash
streamlit run main.py
```

然后在浏览器访问 `http://localhost:8501`

## 📁 项目结构

```
stock-campaign-light-open/
├── main.py                  # 主程序（Streamlit界面）
├── config.py                # 配置管理
├── data_fetcher.py          # 数据获取（Tushare + AKShare）
├── hmm_analyzer.py          # HMM分析核心
├── enhanced_analyzer.py     # 增强版三维度分析器
├── market_context_fetcher.py # 市场面数据获取
├── news_analyzer.py         # 信息面新闻分析
├── dna_manager.py           # 股票DNA管理
├── llm_analyzer.py          # OpenAI 兼容 LLM API调用
├── cycle_predictor.py       # 周期预测器
├── db_manager.py            # 数据库管理
├── model_optimizer.py       # 模型优化器
├── history_manager.py       # 历史记录管理
├── requirements.txt         # 依赖列表
├── data/               # 数据缓存目录
├── dna_history/        # 股票DNA保存目录
└── analysis_history/   # 分析历史保存目录
```

## 🔍 使用方法

### 单股分析

1. 在侧边栏配置 API 信息
   - Tushare Pro Token
   - LLM API Key、Base URL、Model
   - 可使用"快速选择预设"自动填充
2. 选择"单股分析"
3. 输入股票代码（如：600519）
4. 点击"开始分析"

首次分析时会自动生成该股票的DNA并保存，后续分析会直接使用已保存的DNA。

### 批量分析

1. 选择"批量分析"
2. 输入多个股票代码（用逗号或换行分隔）
3. 点击"批量分析"
4. 查看汇总结果

建议批量分析时先不使用 LLM，筛选出重点股票后再单独使用 AI 分析。

### 增强三维度分析

增强版分析整合三个维度，提供更全面的判断：

1. **技术面**: HMM模型分析个股阶段
2. **市场面**: 对比大盘/板块走势、计算相对强度
3. **信息面**: 搜索最新新闻、分析市场情绪

使用步骤：
1. 选择"🔥 增强分析"
2. 输入股票代码和所属行业（可选）
3. 选择要分析的面（市场面/信息面）
4. 点击"开始增强分析"

> 💡 **提示**: 信息面分析需要配置 Tavily API Key（在侧边栏"📰 Tavily API 配置"中输入）

## 📊 阶段说明

| 阶段 | 特征 | 操作建议 |
|------|------|----------|
| 🟢 **吸筹期** | 低位震荡，成交量温和放大 | 关注，可逐步建仓 |
| 🔵 **拉升期** | 快速上涨，成交量明显放大 | 持有或适当加仓 |
| 🔴 **派发期** | 高位震荡，放量滞涨 | 逐步减仓 |
| ⚪ **观望期** | 无明显趋势，成交量萎缩 | 保持观望 |

## ⚙️ 配置说明

API 配置保存在 `user_config.json` 文件中，请勿将此文件提交到GitHub。

```json
{
  "tushare_token": "your_token_here",
  "openai_api_key": "your_key_here",
  "openai_base_url": "https://api.openai.com/v1",
  "openai_model": "gpt-3.5-turbo"
}
```

## 📝 数据存储

- **股票DNA**: 保存在 `dna_history/` 目录，以JSON格式存储
- **分析历史**: 保存在 `analysis_history/` 目录
- **数据缓存**: 临时数据保存在 `data/` 目录

## 🔄 与原版的区别

| 功能 | 原版 | Open 版本 |
|------|------|-----------|
| LLM API | 仅支持 Kimi | 支持**任何 OpenAI 兼容 API** |
| 配置项 | Kimi API Key | API Key + Base URL + Model |
| 预设 | 无 | 内置 14+ 种服务商预设 |
| 适用场景 | Kimi 用户 | 多平台/本地部署/任意 OpenAI 兼容服务 |

## ⚠️ 免责声明

本系统仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。

## 📝 版本更新日志

### v2.0.0 (2026-02-23)

#### ✨ 新功能
- **📰 Tavily API 配置**: 新增 Tavily API Key 配置项，支持信息面新闻分析
- **⚡ 延迟导入优化**: tushare、akshare、hmmlearn 改为延迟导入，大幅提升启动速度
- **📦 本地包缓存**: 新增 packages/ 目录，支持离线快速安装依赖
- **🔧 安装脚本**: 新增 install_deps.sh 一键安装脚本

#### 🐛 修复
- **HMM 模型加载**: 修复从 DNA 加载模型时 "not fitted yet" 错误
- **变量作用域**: 修复保存配置时 "tavily_api_key is not defined" 错误
- **scipy 安装**: 解决 macOS 上 scipy 安装超时和元数据文件问题

#### 📚 文档
- **全新 SETUP.md**: 详细的安装指南，包含故障排查
- **常见问题**: 新增 10+ 个常见问题及解决方案
- **故障排查**: 新增排查章节，包含诊断命令

#### 🔨 改进
- 优化项目结构，新增增强分析模块
- 改进市场面数据获取稳定性
- 优化新闻分析模块的错误处理

---

## 📄 License

MIT License
