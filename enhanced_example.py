"""
增强版分析使用示例
展示如何使用新的三维度分析功能
"""
import os
from data_fetcher import DataFetcher
from dna_manager import DNAManager
from enhanced_analyzer import EnhancedStockAnalyzer


def main():
    """增强版分析示例"""
    
    # 配置API密钥
    TUSHARE_TOKEN = "your_tushare_token"
    OPENAI_API_KEY = "your_openai_api_key"
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # 或其他兼容API
    OPENAI_MODEL = "gpt-3.5-turbo"
    
    # Tavily API Key（免费版，已内置）
    # 如需使用自己的密钥：
    # TAVILY_API_KEY = "tvly-dev-xxxxx"
    
    # 初始化增强版分析器
    analyzer = EnhancedStockAnalyzer(
        tushare_token=TUSHARE_TOKEN,
        openai_api_key=OPENAI_API_KEY,
        openai_base_url=OPENAI_BASE_URL,
        openai_model=OPENAI_MODEL,
        # tavily_api_key=TAVILY_API_KEY  # 可选，默认使用内置密钥
    )
    
    # 初始化数据获取器
    data_fetcher = DataFetcher(tushare_token=TUSHARE_TOKEN)
    dna_manager = DNAManager()
    
    # 分析股票
    stock_code = "600519"  # 茅台
    stock_name = "贵州茅台"
    industry = "白酒"
    
    print(f"=" * 60)
    print(f"增强版分析: {stock_name}({stock_code})")
    print(f"=" * 60)
    
    # 1. 获取股票数据
    print("\n📥 获取股票数据...")
    df, source = data_fetcher.fetch_stock_data(stock_code, days=500)
    
    if df is None:
        print("❌ 获取数据失败")
        return
    
    print(f"✅ 数据来源: {source}, 共{len(df)}天")
    
    # 2. 加载或生成DNA
    dna = dna_manager.load_dna(stock_code)
    if dna:
        print(f"✅ 已加载DNA: {dna.created_at[:10]}")
    else:
        print("📝 首次分析，将生成DNA")
    
    # 3. 执行增强版分析
    result = analyzer.analyze(
        stock_code=stock_code,
        stock_name=stock_name,
        stock_df=df,
        dna=dna,
        industry=industry,
        use_market_context=True,  # 启用市场面分析
        use_news=True             # 启用信息面分析
    )
    
    # 4. 打印结果
    print("\n" + "=" * 60)
    print("📊 分析结果")
    print("=" * 60)
    
    print(f"\n🔹 技术面")
    print(f"   阶段: {result.technical_stage}")
    print(f"   置信度: {result.technical_confidence:.1%}")
    
    print(f"\n🔹 市场面")
    if result.market_context:
        print(f"   大盘趋势: {result.market_context.get('大盘趋势', {})}")
        print(f"   相对大盘强度: {result.relative_strength.get('vs_index', 0):+.1%}")
        print(f"   市场情绪: {result.market_context.get('市场情绪', '未知')}")
    
    print(f"\n🔹 信息面")
    if result.news_analysis:
        print(f"   新闻情绪: {result.news_analysis.get('新闻情绪', '未知')}")
        print(f"   情绪分数: {result.news_analysis.get('情绪分数', 0)}")
        print(f"   关键事件: {result.news_analysis.get('关键事件', [])}")
        print(f"   风险信号: {result.news_analysis.get('风险信号', [])}")
        print(f"   机会信号: {result.news_analysis.get('机会信号', [])}")
    
    print(f"\n🔹 综合判断")
    print(f"   最终阶段: {result.final_stage}")
    print(f"   置信度: {result.final_confidence:.1%}")
    print(f"   分析: {result.comprehensive_analysis}")
    print(f"   操作建议: {result.recommendation}")
    print(f"   风险等级: {result.risk_level}")
    
    # 5. 保存DNA
    if dna is None:
        # 需要重新训练获取DNA
        from hmm_analyzer import HMMAnalyzer
        hmm = HMMAnalyzer()
        dna = hmm.train(df)
        dna.stock_name = stock_name
        dna_manager.save_dna(stock_code, dna)
        print(f"\n💾 DNA已保存")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
