"""
Stock Campaign Light Open - 主程序
轻量版股票战役分析系统 - OpenAI 兼容版本
支持 OpenAI、Azure、Ollama、Kimi 等任何 OpenAI 兼容 API
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import sqlite3

# 导入模块
from config import save_api_keys, load_api_keys, STAGE_NAMES
from data_fetcher import DataFetcher
from hmm_analyzer import HMMAnalyzer
from dna_manager import DNAManager
from llm_analyzer import OpenAIAnalyzer
from history_manager import HistoryManager
from cycle_predictor import CyclePredictor
from db_manager import StockDatabase
from model_optimizer import ModelOptimizer
from enhanced_analyzer import EnhancedStockAnalyzer

# 页面配置
st.set_page_config(
    page_title="Stock Campaign Light Open",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
config = load_api_keys()
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = bool(
        config.get('tushare_token') and config.get('openai_api_key')
    )
    st.session_state.tushare_token = config.get('tushare_token', '')
    st.session_state.openai_api_key = config.get('openai_api_key', '')
    st.session_state.openai_base_url = config.get('openai_base_url', 'https://api.openai.com/v1')
    st.session_state.openai_model = config.get('openai_model', 'gpt-3.5-turbo')

# 确保 api_configured 始终与当前配置同步
st.session_state.api_configured = bool(
    st.session_state.get('tushare_token') and st.session_state.get('openai_api_key')
)

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# 初始化管理器
dna_manager = DNAManager()
history_manager = HistoryManager()
cycle_predictor = CyclePredictor()
stock_db = StockDatabase()
model_optimizer = ModelOptimizer()

# 初始化增强分析器（延迟初始化，需要API配置）
enhanced_analyzer = None


# ============ 侧边栏 ============
with st.sidebar:
    st.title("⚙️ 配置")
    
    # API配置
    st.subheader("API密钥设置")
    
    tushare_token = st.text_input(
        "Tushare Pro Token",
        value=st.session_state.tushare_token,
        type="password",
        help="在 https://tushare.pro/register 注册获取"
    )
    
    st.divider()
    st.markdown("**🤖 LLM API 配置 (OpenAI 兼容)**")
    
    openai_api_key = st.text_input(
        "API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="你的 API Key，支持 OpenAI、Kimi、Ollama 等"
    )
    
    openai_base_url = st.text_input(
        "Base URL",
        value=st.session_state.openai_base_url,
        help="""API 基础地址，支持任何 OpenAI 兼容的服务:
- OpenAI: https://api.openai.com/v1
- Kimi: https://api.moonshot.cn/v1
- Ollama: http://localhost:11434/v1
- DeepSeek: https://api.deepseek.com/v1
- SiliconFlow: https://api.siliconflow.cn/v1
- Azure: https://your-resource.openai.azure.com/openai/deployments/your-deployment
- Groq: https://api.groq.com/openai/v1
- 以及其他任何 OpenAI 兼容 API"""
    )
    
    openai_model = st.text_input(
        "Model",
        value=st.session_state.openai_model,
        help="""模型名称，例如:
- gpt-3.5-turbo, gpt-4, gpt-4o
- moonshot-v1-8k, moonshot-v1-32k (Kimi)
- llama2, mistral, qwen (Ollama)
- deepseek-chat, deepseek-coder (DeepSeek)
- 以及其他任何模型名称"""
    )
    
    # 快速选择预设
    preset_configs = OpenAIAnalyzer.get_preset_configs()
    preset_options = ["自定义"] + list(preset_configs.keys())
    
    preset = st.selectbox(
        "快速选择预设",
        preset_options,
        help="选择预设自动填充 Base URL 和 Model"
    )
    
    if preset != "自定义" and preset in preset_configs:
        config = preset_configs[preset]
        st.session_state.openai_base_url = config["base_url"]
        st.session_state.openai_model = config["model"]
        openai_base_url = config["base_url"]
        openai_model = config["model"]
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存密钥", use_container_width=True):
            save_api_keys(tushare_token, openai_api_key, openai_base_url, openai_model)
            st.session_state.tushare_token = tushare_token
            st.session_state.openai_api_key = openai_api_key
            st.session_state.openai_base_url = openai_base_url
            st.session_state.openai_model = openai_model
            st.session_state.api_configured = bool(tushare_token and openai_api_key)
            st.success("已保存")
    
    with col2:
        if st.button("🧪 测试连接", use_container_width=True):
            if openai_api_key:
                with st.spinner("测试 LLM API..."):
                    llm = OpenAIAnalyzer(openai_api_key, openai_base_url, openai_model)
                    if llm.test_connection():
                        st.success(f"{openai_model} 连接正常")
                    else:
                        st.error("连接失败，请检查配置")
            else:
                st.warning("请先输入 API Key")
    
    st.divider()
    
    # 导航
    st.subheader("导航")
    page = st.radio(
        "选择功能",
        ["🏠 首页", "📊 单股分析", "🔥 增强分析", "📈 批量分析", "🧬 DNA管理", "📜 历史记录", "🎯 预测验证"]
    )
    
    st.divider()
    
    # 数据缓存统计
    st.subheader("💾 数据缓存")
    fetcher = DataFetcher(st.session_state.tushare_token)
    cache_stats = fetcher.get_cache_stats()
    st.metric("缓存股票数", cache_stats['stock_count'])
    st.metric("总记录数", cache_stats['total_records'])
    st.metric("数据库大小", f"{cache_stats['db_size_mb']} MB")
    
    # 预测统计
    st.subheader("📊 预测统计")
    pred_stats = stock_db.get_cache_stats()
    st.metric("预测记录", pred_stats['prediction_count'])
    st.metric("已验证", pred_stats['accuracy_count'])
    
    if st.button("🗑️ 清除缓存", use_container_width=True):
        fetcher.clear_cache()
        st.success("缓存已清除")
        st.rerun()
    
    st.divider()
    
    # 统计
    st.subheader("📊 分析统计")
    stats = history_manager.get_statistics()
    st.metric("分析次数", stats['total_records'])
    st.metric("分析股票数", stats['unique_stocks'])


# ============ 页面内容 ============

if page == "🏠 首页":
    st.title("📈 Stock Campaign Light Open")
    st.markdown("""
    ### 轻量版股票战役分析系统 - OpenAI 兼容版本
    
    本系统基于 **HMM（隐马尔可夫模型）** 和 **OpenAI 兼容的 LLM API**，帮助您识别股票的四个阶段：
    
    | 阶段 | 特征 | 操作建议 |
    |------|------|----------|
    | 🟢 **吸筹期** | 低位震荡，成交量温和放大 | 关注，可逐步建仓 |
    | 🔵 **拉升期** | 快速上涨，成交量明显放大 | 持有或适当加仓 |
    | 🔴 **派发期** | 高位震荡，放量滞涨 | 逐步减仓 |
    | ⚪ **观望期** | 无明显趋势，成交量萎缩 | 保持观望 |
    
    ### 主要功能
    
    - **📊 单股分析**：深度分析单只股票，生成DNA并判断当前阶段
    - **📈 批量分析**：同时分析多只股票，快速筛选
    - **🧬 DNA管理**：查看和管理已生成的股票DNA
    - **📜 历史记录**：查看过往分析记录
    
    ### 开始使用
    
    1. 在左侧配置 **Tushare Pro Token** 和 **LLM API** 信息
    2. 支持 OpenAI、Kimi、Ollama 等任何 OpenAI 兼容的 API
    3. 点击 **单股分析** 或 **批量分析** 开始分析
    
    ### 支持的 LLM 服务
    
    本系统支持**任何 OpenAI 兼容的 API**，包括但不限于：
    
    | 服务商 | Base URL | 模型示例 |
    |--------|----------|----------|
    | **OpenAI** | `https://api.openai.com/v1` | `gpt-3.5-turbo`, `gpt-4`, `gpt-4o` |
    | **Kimi (Moonshot)** | `https://api.moonshot.cn/v1` | `moonshot-v1-8k`, `moonshot-v1-32k` |
    | **Ollama (本地)** | `http://localhost:11434/v1` | `llama2`, `mistral`, `qwen` |
    | **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-coder` |
    | **SiliconFlow** | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |
    | **Groq** | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
    | **Azure OpenAI** | `https://your-resource.openai.azure.com/openai/deployments/your-deployment` | `gpt-35-turbo` |
    | **Together AI** | `https://api.together.xyz/v1` | `meta-llama/Llama-3.1-8B-Instruct-Turbo` |
    | **Fireworks** | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p1-8b-instruct` |
    | **LocalAI** | `http://localhost:8080/v1` | 任何本地模型 |
    | **vLLM** | `http://localhost:8000/v1` | 任何本地模型 |
    
    只要 API 遵循 OpenAI 的接口格式，都可以直接使用！
    """)
    
    # 显示已保存的DNA
    st.subheader("🧬 已保存的股票DNA")
    dna_list = dna_manager.list_all_dna()
    if dna_list:
        dna_df = pd.DataFrame([
            {
                '股票代码': code,
                '股票名称': info.get('stock_name', ''),
                '数据天数': info.get('total_days', 0),
                '最后更新': info.get('updated_at', '')[:10]
            }
            for code, info in dna_list.items()
        ])
        st.dataframe(dna_df, use_container_width=True)
    else:
        st.info("暂无保存的DNA，请先进行分析")


elif page == "📊 单股分析":
    st.title("📊 单股分析")
    
    if not st.session_state.api_configured:
        st.warning("⚠️ 请先配置API密钥")
        st.stop()
    
    # 输入股票代码
    stock_input = st.text_input(
        "输入股票代码",
        placeholder="例如: 600519, 000001, 300750"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("🔍 开始分析", use_container_width=True, type="primary")
    
    with col2:
        use_llm = st.checkbox("使用 LLM AI 增强分析", value=True)
    
    if analyze_btn and stock_input:
        stock_codes = [code.strip() for code in stock_input.split(',') if code.strip()]
        
        for stock_code in stock_codes:
            with st.spinner(f"正在分析 {stock_code}..."):
                try:
                    # 1. 获取数据
                    fetcher = DataFetcher(st.session_state.tushare_token)
                    df, source = fetcher.fetch_stock_data(stock_code, days=1000)
                    
                    if df is None:
                        st.error(f"❌ {stock_code}: 无法获取数据")
                        continue
                    
                    stock_name = fetcher.get_stock_name(stock_code)
                    
                    # 2. 检查或生成DNA
                    dna = dna_manager.load_dna(stock_code)
                    analyzer = HMMAnalyzer()
                    
                    if dna is None:
                        st.info(f"🧬 正在为 {stock_code} 生成DNA...")
                        dna = analyzer.train(df)
                        dna.stock_code = stock_code  # 确保使用正确的股票代码
                        dna.stock_name = stock_name
                        dna_manager.save_dna(dna)
                        st.success(f"✅ DNA已生成并保存")
                    else:
                        # 加载已有DNA
                        analyzer._load_from_dna(dna)
                    
                    # 3. HMM分析
                    hmm_result = analyzer.analyze_current_stage(df, dna)
                    
                    # 4. 周期预测
                    with st.spinner("📅 进行周期预测..."):
                        cycle_prediction = cycle_predictor.predict(stock_code, df, hmm_result, dna)
                        # 保存预测到数据库
                        prediction_id = stock_db.save_prediction(cycle_prediction)
                    
                    # 5. LLM分析（如果启用）
                    llm_result = None
                    if use_llm:
                        with st.spinner(f"🤖 {st.session_state.openai_model} 分析中..."):
                            llm = OpenAIAnalyzer(
                                st.session_state.openai_api_key,
                                st.session_state.openai_base_url,
                                st.session_state.openai_model
                            )
                            recent_summary = {
                                'close': round(df['close'].iloc[-1], 2),
                                'return_20d': f"{((df['close'].iloc[-1] / df['close'].iloc[-20]) - 1) * 100:.2f}%",
                                'volume_change': f"{((df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:-5].mean()) - 1) * 100:.2f}%",
                                'price_position': f"{df['price_position'].iloc[-1]:.2%}"
                            }
                            llm_result = llm.analyze_stock(
                                stock_code, stock_name, hmm_result, recent_summary, cycle_prediction.__dict__
                            )
                    
                    # 6. 保存历史
                    full_data = {
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'hmm_result': hmm_result,
                        'llm_result': llm_result,
                        'cycle_prediction': cycle_prediction.__dict__,
                        'prediction_id': prediction_id,
                        'data_source': source,
                        'df_tail': df.tail(20).to_dict('records')
                    }
                    history_manager.save_analysis(
                        stock_code, stock_name, hmm_result, llm_result,
                        df['close'].iloc[-1], full_data
                    )
                    
                    # 显示结果
                    st.subheader(f"📊 {stock_code} {stock_name} 分析结果")
                    
                    # 数据源显示优化
                    source_display = {
                        'local': '📦 本地缓存',
                        'tushare': '💹 Tushare',
                        'akshare': '📈 AKShare',
                        'mixed': '📦+📈 混合',
                        'failed': '❌ 失败'
                    }
                    source_emoji = source_display.get(source, source.upper())
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        stage = hmm_result['current_stage']
                        stage_emoji = {"吸筹期": "🟢", "拉升期": "🔵", "派发期": "🔴", "观望期": "⚪"}
                        st.metric("当前阶段", f"{stage_emoji.get(stage, '')} {stage}")
                    with col2:
                        st.metric("置信度", f"{hmm_result['confidence']:.2%}")
                    with col3:
                        st.metric("数据天数", f"{len(df)}天")
                    with col4:
                        st.metric("数据源", source_emoji)
                    
                    # 显示各阶段概率
                    st.write("**各阶段概率:**")
                    prob_cols = st.columns(4)
                    for i, (stage_name, prob) in enumerate(hmm_result['all_stage_probs'].items()):
                        with prob_cols[i]:
                            st.progress(prob, text=f"{stage_name}: {prob:.1%}")
                    
                    # 显示周期预测信息
                    st.write("---")
                    st.write("**📅 周期预测信息:**")
                    
                    # 显示历史周期统计
                    if cycle_prediction.hist_avg_duration > 0:
                        with st.expander("📊 历史周期统计"):
                            hist_cols = st.columns(5)
                            with hist_cols[0]:
                                st.metric("历史平均", f"{cycle_prediction.hist_avg_duration}天")
                            with hist_cols[1]:
                                st.metric("历史最短", f"{cycle_prediction.hist_min_duration}天")
                            with hist_cols[2]:
                                st.metric("历史最长", f"{cycle_prediction.hist_max_duration}天")
                            with hist_cols[3]:
                                st.metric("当前已持续", f"{cycle_prediction.days_in_cycle}天")
                            with hist_cols[4]:
                                st.metric("历史样本", f"{cycle_prediction.hist_sample_count}个")
                    
                    # 周期时间信息
                    cycle_cols = st.columns(3)
                    with cycle_cols[0]:
                        st.metric("周期开始", cycle_prediction.cycle_start_date)
                    with cycle_cols[1]:
                        st.metric("预计结束", cycle_prediction.cycle_estimated_end)
                    with cycle_cols[2]:
                        st.metric("当前进度", f"{cycle_prediction.cycle_progress:.1%}")
                    
                    # 周期进度条
                    progress_text = f"当前阶段已持续 {cycle_prediction.days_in_cycle} 天"
                    if cycle_prediction.hist_avg_duration > 0:
                        progress_text += f" / 历史平均{cycle_prediction.hist_avg_duration}天"
                    else:
                        progress_text += f" / 预计共{cycle_prediction.estimated_total_days}天"
                    st.progress(cycle_prediction.cycle_progress, text=progress_text)
                    
                    # 价格预测
                    st.write("**💰 价格预测:**")
                    price_cols = st.columns(3)
                    with price_cols[0]:
                        st.metric("当前价格", f"{cycle_prediction.current_price:.2f}")
                    with price_cols[1]:
                        st.metric("周期结束目标价", 
                                 f"{cycle_prediction.price_target_mean:.2f}",
                                 f"{cycle_prediction.price_target_low:.2f} - {cycle_prediction.price_target_high:.2f}")
                    with price_cols[2]:
                        price_change = (cycle_prediction.price_target_mean - cycle_prediction.current_price) / cycle_prediction.current_price * 100
                        st.metric("预期涨跌", f"{price_change:+.1f}%")
                    
                    # 短期预测
                    st.write("**📊 短期预测:**")
                    pred_cols = st.columns(2)
                    with pred_cols[0]:
                        st.write(f"**未来5天:** {cycle_prediction.pred_5d_stage}")
                        st.write(f"价格区间: {cycle_prediction.pred_5d_price_low:.2f} - {cycle_prediction.pred_5d_price_high:.2f}")
                        st.write(f"置信度: {cycle_prediction.pred_5d_confidence:.0%}")
                    with pred_cols[1]:
                        st.write(f"**未来20天:** {cycle_prediction.pred_20d_stage}")
                        st.write(f"价格区间: {cycle_prediction.pred_20d_price_low:.2f} - {cycle_prediction.pred_20d_price_high:.2f}")
                        st.write(f"置信度: {cycle_prediction.pred_20d_confidence:.0%}")
                    
                    # 预测依据
                    with st.expander("📖 预测依据"):
                        st.write(cycle_prediction.reasoning)
                        st.write("**关键指标:**")
                        for key, value in cycle_prediction.key_indicators.items():
                            if isinstance(value, float):
                                st.write(f"- {key}: {value:.4f}")
                            else:
                                st.write(f"- {key}: {value}")
                    
                    # 显示LLM分析结果
                    if llm_result and 'error' not in llm_result:
                        st.write("---")
                        st.write(f"**🤖 {st.session_state.openai_model} AI 分析:**")
                        st.write(f"- **阶段认同**: {llm_result.get('stage_agreement', 'N/A')}")
                        st.write(f"- **分析理由**: {llm_result.get('stage_reasoning', 'N/A')}")
                        st.write(f"- **风险评级**: {llm_result.get('risk_assessment', 'N/A')}")
                        st.write(f"- **操作建议**: {llm_result.get('suggestion', 'N/A')}")
                        if llm_result.get('key_signals'):
                            st.write(f"- **关键信号**: {', '.join(llm_result['key_signals'])}")
                    
                    # 价格走势图
                    st.write("---")
                    st.write("**📈 近期价格走势与成交量:**")
                    chart_df = df.tail(60).copy()
                    chart_df['日期'] = chart_df['date']
                    
                    # 使用两列布局显示价格和成交量
                    price_tab, volume_tab = st.tabs(["价格走势", "成交量"])
                    
                    with price_tab:
                        st.line_chart(chart_df.set_index('日期')[['close', 'ma20']], use_container_width=True)
                    
                    with volume_tab:
                        # 计算成交量MA5和MA20
                        if 'volume_ma5' in chart_df.columns and 'volume_ma20' in chart_df.columns:
                            st.bar_chart(chart_df.set_index('日期')[['volume']], use_container_width=True)
                        else:
                            st.bar_chart(chart_df.set_index('日期')[['volume']], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ {stock_code} 分析失败: {str(e)}")


elif page == "🔥 增强分析":
    st.title("🔥 增强版三维度分析")
    
    st.markdown("""
    ### 📊 技术面 + 📈 市场面 + 📰 信息面 = 综合判断
    
    增强版分析整合三个维度：
    - **技术面**: HMM模型分析个股阶段
    - **市场面**: 对比大盘/板块走势、相对强度
    - **信息面**: Tavily搜索最新新闻、情绪分析
    """)
    
    if not st.session_state.api_configured:
        st.warning("⚠️ 请先配置API密钥（Tushare + LLM API）")
        st.stop()
    
    # 初始化增强分析器
    if enhanced_analyzer is None:
        enhanced_analyzer = EnhancedStockAnalyzer(
            tushare_token=st.session_state.tushare_token,
            openai_api_key=st.session_state.openai_api_key,
            openai_base_url=st.session_state.openai_base_url,
            openai_model=st.session_state.openai_model
        )
    
    # 输入区域
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_input = st.text_input(
            "输入股票代码",
            placeholder="例如: 600519",
            key="enhanced_stock"
        )
    with col2:
        industry_input = st.text_input(
            "所属行业（可选）",
            placeholder="例如: 白酒、新能源",
            key="enhanced_industry"
        )
    
    # 分析选项
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        use_market = st.checkbox("市场面分析", value=True)
    with col2:
        use_news = st.checkbox("信息面分析", value=True)
    with col3:
        st.caption("信息面分析使用Tavily免费搜索API")
    
    if st.button("🔥 开始增强分析", type="primary", use_container_width=True):
        if not stock_input:
            st.warning("请输入股票代码")
            st.stop()
        
        stock_code = stock_input.strip()
        industry = industry_input.strip()
        
        # 创建进度区域
        progress_area = st.empty()
        status_area = st.empty()
        
        try:
            # 1. 获取数据
            status_area.info("📥 正在获取股票数据...")
            fetcher = DataFetcher(st.session_state.tushare_token)
            df, source = fetcher.fetch_stock_data(stock_code, days=500)
            
            if df is None:
                st.error(f"❌ 无法获取 {stock_code} 的数据")
                st.stop()
            
            stock_name = fetcher.get_stock_name(stock_code)
            status_area.success(f"✅ 数据获取完成: {stock_name} ({len(df)}天)")
            
            # 2. 加载或生成DNA
            dna = dna_manager.load_dna(stock_code)
            if dna is None:
                status_area.info("🧬 正在生成DNA...")
                from hmm_analyzer import HMMAnalyzer
                hmm = HMMAnalyzer()
                dna = hmm.train(df)
                dna.stock_name = stock_name
                dna_manager.save_dna(dna)
                status_area.success("✅ DNA已生成并保存")
            else:
                status_area.info(f"📦 已加载DNA: {dna.created_at[:10]}")
            
            # 3. 执行增强分析（分步骤）
            status_area.info("🔍 开始三维度分析...")
            
            # 技术分析
            if use_market:
                status_area.info("📈 正在进行市场面分析...")
            if use_news:
                status_area.info("📰 正在进行信息面分析（Tavily搜索）...")
            
            result = enhanced_analyzer.analyze(
                stock_code=stock_code,
                stock_name=stock_name,
                stock_df=df,
                dna=dna,
                industry=industry,
                use_market_context=use_market,
                use_news=use_news
            )
            
            status_area.empty()  # 清除状态信息
                
                # 4. 显示结果
                st.divider()
                st.subheader(f"📊 {stock_code} {stock_name} - 增强分析结果")
                
                # 综合判断卡片
                stage_emoji = {"吸筹期": "🟢", "拉升期": "🔵", "派发期": "🔴", "观望期": "⚪"}
                stage_color = {"吸筹期": "green", "拉升期": "blue", "派发期": "red", "观望期": "gray"}
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {'#1a1a2e' if result.final_stage == '拉升期' else '#2d132c' if result.final_stage == '派发期' else '#1e3a3a' if result.final_stage == '吸筹期' else '#2c2c2c'}; color: white;">
                    <h2 style="margin: 0;">{stage_emoji.get(result.final_stage, '')} 综合判断: {result.final_stage}</h2>
                    <p style="font-size: 18px; margin: 10px 0;">置信度: {result.final_confidence:.1%} | 风险等级: {result.risk_level.upper()}</p>
                    <p style="font-size: 16px; margin: 10px 0;"><strong>操作建议:</strong> {result.recommendation}</p>
                    <p style="font-size: 14px; color: #cccccc;">{result.comprehensive_analysis}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 三维度详情
                tab1, tab2, tab3 = st.tabs(["📊 技术面", "📈 市场面", "📰 信息面"])
                
                with tab1:
                    st.write("**HMM模型分析**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("技术阶段", result.technical_stage)
                    with col2:
                        st.metric("技术置信度", f"{result.technical_confidence:.1%}")
                    
                    # 各阶段概率
                    st.write("**各阶段概率:**")
                    probs = result.technical_details.get('all_stage_probs', {})
                    prob_cols = st.columns(4)
                    for i, (stage, prob) in enumerate(probs.items()):
                        with prob_cols[i]:
                            st.progress(prob, text=f"{stage}: {prob:.1%}")
                    
                    # 特征分析
                    with st.expander("特征分析详情"):
                        features = result.technical_details.get('feature_analysis', {})
                        for name, value in features.items():
                            st.write(f"- {name}: {value:.4f}")
                
                with tab2:
                    if result.market_context:
                        st.write("**市场环境**")
                        
                        # 大盘趋势
                        index_trends = result.market_context.get('大盘趋势', {})
                        if index_trends:
                            st.write("**大盘趋势:**")
                            for idx, trend in list(index_trends.items())[:3]:
                                st.write(f"- {idx}: {trend}")
                        
                        # 相对强度
                        st.write("**相对强度:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            rs_index = result.relative_strength.get('vs_index', 0)
                            st.metric("相对大盘", f"{rs_index:+.1%}", 
                                     delta="强于大盘" if rs_index > 0 else "弱于大盘")
                        with col2:
                            rs_sector = result.relative_strength.get('vs_sector', 0)
                            st.metric("相对板块", f"{rs_sector:+.1%}",
                                     delta="强于板块" if rs_sector > 0 else "弱于板块")
                        
                        # 市场情绪
                        sentiment = result.market_context.get('市场情绪', '未知')
                        sentiment_emoji = {
                            '极度乐观': '😄', '乐观': '🙂', '中性': '😐',
                            '恐慌': '😟', '极度恐慌': '😱'
                        }
                        st.write(f"**市场情绪:** {sentiment_emoji.get(sentiment, '')} {sentiment}")
                    else:
                        st.info("未启用市场面分析")
                
                with tab3:
                    if result.news_analysis and result.news_analysis.get('股票代码'):
                        st.write("**新闻情绪分析**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment = result.news_analysis.get('新闻情绪', 'unknown')
                            sentiment_display = {'positive': '积极', 'negative': '消极', 'neutral': '中性'}
                            st.metric("新闻情绪", sentiment_display.get(sentiment, sentiment))
                        with col2:
                            st.metric("情绪分数", result.news_analysis.get('情绪分数', 0))
                        
                        # 关键事件
                        key_events = result.news_analysis.get('关键事件', [])
                        if key_events:
                            st.write("**关键事件:**")
                            for event in key_events:
                                st.write(f"- {event}")
                        
                        # 风险与机会
                        col1, col2 = st.columns(2)
                        with col1:
                            risks = result.news_analysis.get('风险信号', [])
                            if risks:
                                st.write("**⚠️ 风险信号:**")
                                for r in risks:
                                    st.write(f"- {r}")
                        with col2:
                            opportunities = result.news_analysis.get('机会信号', [])
                            if opportunities:
                                st.write("**✅ 机会信号:**")
                                for o in opportunities:
                                    st.write(f"- {o}")
                        
                        # 综合摘要
                        with st.expander("新闻摘要"):
                            st.write(result.news_analysis.get('综合摘要', '暂无'))
                    else:
                        st.info("未启用信息面分析")
                
                # 保存历史
                full_data = {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'enhanced_analysis': result.to_dict(),
                    'data_source': source
                }
                history_manager.save_analysis(
                    stock_code, stock_name,
                    {'current_stage': result.technical_stage, 'confidence': result.technical_confidence},
                    {'stage_agreement': result.final_stage, 'suggestion': result.recommendation},
                    df['close'].iloc[-1], full_data
                )
                
                st.success("✅ 分析结果已保存到历史记录")
                
            except Exception as e:
                st.error(f"❌ 分析失败: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


elif page == "📈 批量分析":
    st.title("📈 批量分析")
    
    if not st.session_state.api_configured:
        st.warning("⚠️ 请先配置API密钥")
        st.stop()
    
    st.markdown("输入多个股票代码，用逗号或换行分隔")
    
    stock_input = st.text_area(
        "股票代码列表",
        placeholder="例如:\n600519\n000001\n300750",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        batch_btn = st.button("🚀 批量分析", use_container_width=True, type="primary")
    
    with col2:
        use_llm_batch = st.checkbox("使用 LLM AI（批量分析时较慢）", value=False)
        st.caption("建议: 批量分析时先不使用LLM，筛选出重点股票后再单独用LLM分析")
    
    if batch_btn and stock_input:
        # 解析股票代码
        stock_codes = []
        for line in stock_input.split('\n'):
            for code in line.split(','):
                code = code.strip()
                if code:
                    stock_codes.append(code)
        
        stock_codes = list(set(stock_codes))  # 去重
        
        if len(stock_codes) > 20:
            st.warning(f"⚠️ 一次最多分析20只股票，当前 {len(stock_codes)} 只，将只分析前20只")
            stock_codes = stock_codes[:20]
        
        st.write(f"📋 即将分析 {len(stock_codes)} 只股票: {', '.join(stock_codes)}")
        
        # 批量分析
        results = []
        progress = st.progress(0)
        
        fetcher = DataFetcher(st.session_state.tushare_token)
        
        for i, stock_code in enumerate(stock_codes):
            progress.progress((i + 1) / len(stock_codes), f"分析 {stock_code}...")
            
            try:
                # 获取数据
                df, source = fetcher.fetch_stock_data(stock_code, days=1000)
                if df is None:
                    results.append({
                        '股票代码': stock_code,
                        '状态': '❌ 数据获取失败',
                        '阶段': '-',
                        '置信度': '-'
                    })
                    continue
                
                stock_name = fetcher.get_stock_name(stock_code)
                
                # 检查DNA
                dna = dna_manager.load_dna(stock_code)
                analyzer = HMMAnalyzer()
                
                if dna is None:
                    dna = analyzer.train(df)
                    dna.stock_name = stock_name
                    dna_manager.save_dna(dna)
                else:
                    analyzer._load_from_dna(dna)
                
                # 分析
                hmm_result = analyzer.analyze_current_stage(df)
                
                results.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name,
                    '状态': '✅ 成功',
                    '阶段': hmm_result['current_stage'],
                    '置信度': f"{hmm_result['confidence']:.2%}",
                    '数据源': source
                })
                
                # 保存历史
                history_manager.save_analysis(
                    stock_code, stock_name, hmm_result, None,
                    df['close'].iloc[-1], {
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'hmm_result': hmm_result,
                        'data_source': source
                    }
                )
                
            except Exception as e:
                results.append({
                    '股票代码': stock_code,
                    '状态': f'❌ 错误: {str(e)[:30]}',
                    '阶段': '-',
                    '置信度': '-'
                })
        
        progress.empty()
        
        # 显示结果表格
        st.subheader("📊 分析结果汇总")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # 阶段分布统计
        if results:
            stage_counts = {}
            for r in results:
                if '阶段' in r and r['阶段'] != '-':
                    stage_counts[r['阶段']] = stage_counts.get(r['阶段'], 0) + 1
            
            if stage_counts:
                st.write("**阶段分布:**")
                cols = st.columns(len(stage_counts))
                for i, (stage, count) in enumerate(stage_counts.items()):
                    with cols[i]:
                        st.metric(stage, f"{count}只")


elif page == "🧬 DNA管理":
    st.title("🧬 DNA管理")
    
    # 列出所有DNA
    dna_list = dna_manager.list_all_dna()
    
    if not dna_list:
        st.info("暂无保存的股票DNA")
    else:
        st.write(f"共保存了 **{len(dna_list)}** 个股票DNA")
        
        for stock_code, info in dna_list.items():
            with st.expander(f"📋 {stock_code} {info.get('stock_name', '')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"- **数据天数**: {info.get('total_days', 0)}")
                    st.write(f"- **创建时间**: {info.get('created_at', '')}")
                    st.write(f"- **最后更新**: {info.get('updated_at', '')}")
                    
                    # 阶段分布
                    st.write("**历史阶段分布:**")
                    dist = info.get('stage_distribution', {})
                    for stage, ratio in dist.items():
                        st.progress(ratio, text=f"{stage}: {ratio:.1%}")
                
                with col2:
                    if st.button("🗑️ 删除", key=f"del_{stock_code}"):
                        dna_manager.delete_dna(stock_code)
                        st.success(f"已删除 {stock_code}")
                        st.rerun()
                    
                    if st.button("📊 查看详情", key=f"detail_{stock_code}"):
                        dna = dna_manager.load_dna(stock_code)
                        if dna:
                            st.json({
                                'stage_templates': dna.stage_templates,
                                'transition_matrix': dna.transition_matrix[:2]  # 只显示部分
                            })


elif page == "📜 历史记录":
    st.title("📜 分析历史")
    
    # 筛选
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_code = st.text_input("筛选股票代码（留空显示所有）")
    with col2:
        limit = st.number_input("显示条数", min_value=10, max_value=100, value=50)
    
    # 获取历史
    history = history_manager.get_history(
        stock_code=filter_code if filter_code else None,
        limit=limit
    )
    
    if not history:
        st.info("暂无分析历史")
    else:
        st.write(f"共 **{len(history)}** 条记录")
        
        for record in history:
            with st.expander(
                f"{record['analysis_time'][:16]} - "
                f"{record['stock_code']} {record['stock_name']} - "
                f"{record['hmm_stage']}"
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"- **股票**: {record['stock_code']} {record['stock_name']}")
                    st.write(f"- **分析时间**: {record['analysis_time']}")
                    st.write(f"- **HMM判断**: {record['hmm_stage']} (置信度: {record['hmm_confidence']:.2%})")
                    st.write(f"- **收盘价**: {record['close_price']:.2f}")
                    
                    if record.get('llm_stage_agreement'):
                        st.write(f"- **LLM认同**: {record['llm_stage_agreement']}")
                    if record.get('llm_suggestion'):
                        st.write(f"- **LLM建议**: {record['llm_suggestion']}")
                
                with col2:
                    if st.button("🗑️ 删除", key=f"del_hist_{record['record_id']}"):
                        history_manager.delete_record(record['record_id'])
                        st.success("已删除")
                        st.rerun()
                    
                    if st.button("📄 查看详情", key=f"detail_hist_{record['record_id']}"):
                        detail = history_manager.get_detail(record['record_id'])
                        if detail:
                            st.json(detail['hmm_result'])


elif page == "🎯 预测验证":
    st.title("🎯 预测验证与模型优化")
    
    st.markdown("""
    通过验证历史预测的准确性，不断优化模型参数。
    系统会自动对比预测价格与实际价格，计算预测准确率。
    """)
    
    # 预测统计概览
    st.subheader("📊 预测准确性统计")
    accuracy_stats = stock_db.get_prediction_accuracy_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均准确率评分", f"{accuracy_stats['avg_accuracy_score']:.2%}")
    with col2:
        st.metric("平均价格误差率", f"{accuracy_stats['avg_price_error_rate']:.2%}")
    with col3:
        st.metric("阶段预测正确率", f"{accuracy_stats['stage_correct_rate']:.2%}")
    with col4:
        st.metric("已验证记录数", accuracy_stats['total_evaluations'])
    
    st.divider()
    
    # 验证预测
    st.subheader("🔍 验证待验证的预测")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        verify_stock = st.text_input("股票代码", placeholder="例如: 600519")
    with col2:
        verify_type = st.selectbox("预测类型", ["5天", "20天", "周期结束"])
    
    if st.button("🔍 查找可验证的预测", type="primary"):
        if verify_stock:
            predictions = stock_db.get_predictions(verify_stock, limit=20)
            if predictions:
                st.write(f"找到 **{len(predictions)}** 条预测记录")
                
                for pred in predictions:
                    pred_id = pred['id']
                    analysis_date = pred['analysis_date']
                    
                    # 根据验证类型确定预测日期和价格
                    if verify_type == "5天":
                        target_date = (pd.to_datetime(analysis_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
                        pred_price_low = pred['pred_5d_price_low']
                        pred_price_high = pred['pred_5d_price_high']
                        pred_stage = pred['pred_5d_stage']
                        pred_type_code = '5d'
                    elif verify_type == "20天":
                        target_date = (pd.to_datetime(analysis_date) + pd.Timedelta(days=20)).strftime('%Y-%m-%d')
                        pred_price_low = pred['pred_20d_price_low']
                        pred_price_high = pred['pred_20d_price_high']
                        pred_stage = pred['pred_20d_stage']
                        pred_type_code = '20d'
                    else:  # 周期结束
                        target_date = pred['cycle_estimated_end']
                        pred_price_low = pred['price_target_low']
                        pred_price_high = pred['price_target_high']
                        pred_stage = pred['current_stage']
                        pred_type_code = 'cycle_end'
                    
                    with st.expander(f"📅 {analysis_date} 的预测 (预测 {target_date})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**预测阶段:** {pred_stage}")
                            st.write(f"**预测价格区间:** {pred_price_low:.2f} - {pred_price_high:.2f}")
                            st.write(f"**预测时价格:** {pred['current_price']:.2f}")
                        with col2:
                            actual_price = st.number_input(
                                f"实际价格 ({target_date})", 
                                min_value=0.0, 
                                value=pred_price_low,
                                key=f"price_{pred_id}_{pred_type_code}"
                            )
                            actual_stage = st.selectbox(
                                f"实际阶段",
                                ["吸筹期", "拉升期", "派发期", "观望期"],
                                index=["吸筹期", "拉升期", "派发期", "观望期"].index(pred_stage) if pred_stage in ["吸筹期", "拉升期", "派发期", "观望期"] else 0,
                                key=f"stage_{pred_id}_{pred_type_code}"
                            )
                        
                        if st.button(f"✅ 提交验证", key=f"verify_{pred_id}_{pred_type_code}"):
                            result = stock_db.evaluate_prediction(
                                pred_id, verify_stock, pred_type_code,
                                actual_price, actual_stage, analysis_date
                            )
                            if 'error' not in result:
                                st.success(f"验证完成！准确率评分: {result['accuracy_score']:.2%}")
                                if result['stage_correct']:
                                    st.success("✅ 阶段预测正确")
                                else:
                                    st.error(f"❌ 阶段预测错误 (预测: {result['predicted_stage']}, 实际: {result['actual_stage']})")
                                st.info(f"价格误差率: {result['price_error_rate']:.2%}")
                            else:
                                st.error(result['error'])
            else:
                st.info("暂无预测记录")
        else:
            st.warning("请输入股票代码")
    
    st.divider()
    
    # 显示详细准确性统计
    st.subheader("📈 各股票预测准确性")
    if st.button("📊 生成统计报告"):
        # 获取所有有预测记录的股票
        conn = sqlite3.connect(stock_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT stock_code FROM cycle_predictions")
        stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if stocks:
            stats_data = []
            for stock in stocks:
                stats = stock_db.get_prediction_accuracy_stats(stock)
                if stats['total_evaluations'] > 0:
                    stats_data.append({
                        '股票代码': stock,
                        '验证次数': stats['total_evaluations'],
                        '平均准确率': f"{stats['avg_accuracy_score']:.2%}",
                        '阶段正确率': f"{stats['stage_correct_rate']:.2%}",
                        '平均价格误差': f"{stats['avg_price_error_rate']:.2%}"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # 可视化
                st.write("**准确率分布:**")
                chart_data = pd.DataFrame({
                    '股票': [s['股票代码'] for s in stats_data],
                    '准确率': [float(s['平均准确率'].rstrip('%'))/100 for s in stats_data]
                })
                st.bar_chart(chart_data.set_index('股票'))
            else:
                st.info("暂无已验证的预测数据")
        else:
            st.info("暂无预测记录")
    
    st.divider()
    
    # 模型优化
    st.subheader("🔧 模型参数优化")
    st.markdown("""
    基于历史预测准确性数据，自动优化模型参数。
    系统会分析预测误差，建议调整各阶段的持续时间等参数。
    """)
    
    if st.button("📊 生成优化报告"):
        report = model_optimizer.get_optimization_report()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("整体预测准确率", f"{report['overall_accuracy']:.2%}")
        with col2:
            st.metric("已验证预测数", report['total_evaluated_predictions'])
        
        # 显示各阶段准确性
        if report['accuracy_by_stage']:
            st.write("**各阶段预测准确性:**")
            accuracy_data = []
            for stage, type_stats in report['accuracy_by_stage'].items():
                for pred_type, stats in type_stats.items():
                    accuracy_data.append({
                        '阶段': stage,
                        '预测类型': pred_type,
                        '准确率': f"{stats['avg_accuracy']:.2%}",
                        '价格误差': f"{stats['avg_price_error']:.2%}",
                        '阶段正确率': f"{stats['stage_correct_rate']:.2%}",
                        '样本数': stats['sample_count']
                    })
            
            if accuracy_data:
                st.dataframe(pd.DataFrame(accuracy_data), use_container_width=True)
        
        # 参数调整建议
        if report['parameter_suggestions']:
            st.write("**🔧 参数调整建议:**")
            for suggestion in report['parameter_suggestions'][:5]:  # 只显示前5条
                with st.expander(f"{suggestion['stage']} - {suggestion['parameter']} (置信度: {suggestion['confidence']:.0%})"):
                    st.write(f"**建议:** {suggestion['reason']}")
                    st.write(f"**当前值:** {suggestion['current_value']}")
                    st.write(f"**建议值:** {suggestion['suggested_value']}")
        else:
            st.success("✅ 当前参数表现良好，无需调整")
        
        # 应用调整
        if report['parameter_suggestions']:
            st.write("**应用调整:**")
            dry_run = st.checkbox("仅预览（不实际应用）", value=True)
            
            if st.button("🚀 应用参数调整"):
                result = model_optimizer.apply_adjustments(
                    report['parameter_suggestions'], 
                    dry_run=dry_run
                )
                
                if dry_run:
                    st.info("📋 预览模式 - 以下调整将被应用:")
                else:
                    st.success("✅ 参数调整已应用")
                
                for item in result['applied']:
                    st.write(f"- {item['stage']}: {item['parameter']} = {item['suggested_value']}")
                
                if result['skipped']:
                    st.warning(f"跳过 {len(result['skipped'])} 项调整（置信度不足）")
    
    # 重置参数
    with st.expander("⚠️ 重置参数"):
        st.warning("这将把所有参数重置为默认值")
        if st.button("🔄 重置为默认参数"):
            model_optimizer.reset_to_defaults()
            st.success("✅ 参数已重置为默认值")
