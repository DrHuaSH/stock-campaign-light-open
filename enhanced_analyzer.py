"""
增强版股票分析器 - 整合技术面、市场面、信息面
"""
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas as pd

from hmm_analyzer import HMMAnalyzer, StockDNA
from market_context_fetcher import MarketContextFetcher, MarketContext
from news_analyzer import NewsAnalyzer, NewsAnalysis
from config import STAGE_NAMES, STAGE_NAMES_EN


@dataclass
class EnhancedAnalysisResult:
    """增强版分析结果"""
    # 基础信息
    stock_code: str
    stock_name: str
    
    # 技术面分析（HMM）
    technical_stage: str
    technical_confidence: float
    technical_details: Dict
    
    # 市场面分析
    market_context: Dict
    relative_strength: Dict[str, float]
    
    # 信息面分析
    news_analysis: Dict
    
    # 综合判断
    final_stage: str
    final_confidence: float
    comprehensive_analysis: str
    
    # 操作建议
    recommendation: str
    risk_level: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            '股票代码': self.stock_code,
            '股票名称': self.stock_name,
            '技术面': {
                '阶段': self.technical_stage,
                '置信度': f"{self.technical_confidence:.1%}",
                '详情': self.technical_details
            },
            '市场面': {
                '市场环境': self.market_context,
                '相对强度': self.relative_strength
            },
            '信息面': self.news_analysis,
            '综合判断': {
                '阶段': self.final_stage,
                '置信度': f"{self.final_confidence:.1%}",
                '分析': self.comprehensive_analysis
            },
            '操作建议': self.recommendation,
            '风险等级': self.risk_level
        }


class EnhancedStockAnalyzer:
    """增强版股票分析器"""
    
    def __init__(
        self,
        tushare_token: str = "",
        openai_api_key: str = "",
        openai_base_url: str = "https://api.openai.com/v1",
        openai_model: str = "gpt-3.5-turbo",
        tavily_api_key: str = ""
    ):
        self.tushare_token = tushare_token
        
        # 初始化各模块
        self.hmm_analyzer = HMMAnalyzer(n_states=4)
        self.market_fetcher = MarketContextFetcher(tushare_token)
        self.news_analyzer = NewsAnalyzer(
            tavily_api_key=tavily_api_key,
            tushare_token=tushare_token,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model
        )
        
        # OpenAI配置（用于综合分析）
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
    
    def analyze(
        self,
        stock_code: str,
        stock_name: str,
        stock_df: pd.DataFrame,
        dna: Optional[StockDNA] = None,
        industry: str = "",
        use_market_context: bool = True,
        use_news: bool = True
    ) -> EnhancedAnalysisResult:
        """
        执行增强版分析
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            stock_df: 股票历史数据
            dna: 股票DNA（可选）
            industry: 所属行业
            use_market_context: 是否使用市场面分析
            use_news: 是否使用信息面分析
        
        Returns:
            EnhancedAnalysisResult
        """
        print(f"\n🔍 开始增强分析: {stock_name}({stock_code})")
        
        # 1. 技术面分析（HMM）
        print("  📊 技术面分析...")
        if dna:
            self.hmm_analyzer._load_from_dna(dna)
            technical_result = self.hmm_analyzer.analyze_current_stage(stock_df, dna)
        else:
            dna = self.hmm_analyzer.train(stock_df)
            technical_result = self.hmm_analyzer.analyze_current_stage(stock_df)
        
        technical_stage = technical_result['current_stage']
        technical_confidence = technical_result['confidence']
        
        # 2. 市场面分析
        market_context = None
        if use_market_context and self.tushare_token:
            print("  📈 市场面分析...")
            try:
                market_context = self.market_fetcher.get_market_context(stock_code, stock_df)
            except Exception as e:
                print(f"    ⚠️ 市场面分析失败: {e}")
        
        # 3. 信息面分析
        news_analysis = None
        if use_news:
            print("  📰 信息面分析...")
            try:
                news_analysis = self.news_analyzer.analyze_stock_news(
                    stock_code, stock_name, industry
                )
            except Exception as e:
                print(f"    ⚠️ 信息面分析失败: {e}")
        
        # 4. 综合分析
        print("  🧠 综合分析...")
        final_result = self._comprehensive_analysis(
            stock_code,
            stock_name,
            technical_result,
            market_context,
            news_analysis
        )
        
        return EnhancedAnalysisResult(
            stock_code=stock_code,
            stock_name=stock_name,
            technical_stage=technical_stage,
            technical_confidence=technical_confidence,
            technical_details=technical_result,
            market_context=market_context.to_dict() if market_context else {},
            relative_strength={
                'vs_index': market_context.relative_strength_vs_index if market_context else 0,
                'vs_sector': market_context.relative_strength_vs_sector if market_context else 0
            },
            news_analysis=news_analysis.to_dict() if news_analysis else {},
            final_stage=final_result['stage'],
            final_confidence=final_result['confidence'],
            comprehensive_analysis=final_result['analysis'],
            recommendation=final_result['recommendation'],
            risk_level=final_result['risk_level']
        )
    
    def _comprehensive_analysis(
        self,
        stock_code: str,
        stock_name: str,
        technical_result: Dict,
        market_context: Optional[MarketContext],
        news_analysis: Optional[NewsAnalysis]
    ) -> Dict:
        """
        综合分析三维度数据，得出最终判断
        """
        technical_stage = technical_result['current_stage']
        technical_confidence = technical_result['confidence']
        
        # 如果没有市场面和信息面，直接返回技术面结果
        if not market_context and not news_analysis:
            return {
                'stage': technical_stage,
                'confidence': technical_confidence,
                'analysis': f'基于技术分析，当前处于{technical_stage}',
                'recommendation': self._get_basic_recommendation(technical_stage),
                'risk_level': 'medium'
            }
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(
            stock_name, technical_result, market_context, news_analysis
        )
        
        # 使用LLM进行综合分析
        if self.openai_api_key:
            try:
                return self._llm_comprehensive_analysis(prompt)
            except Exception as e:
                print(f"    ⚠️ LLM综合分析失败: {e}，使用规则引擎")
        
        # 使用规则引擎综合分析
        return self._rule_comprehensive_analysis(
            technical_stage, technical_confidence, market_context, news_analysis
        )
    
    def _build_analysis_prompt(
        self,
        stock_name: str,
        technical_result: Dict,
        market_context: Optional[MarketContext],
        news_analysis: Optional[NewsAnalysis]
    ) -> str:
        """构建综合分析提示"""
        
        prompt = f"""请综合分析以下股票数据，给出投资判断：

## 股票：{stock_name}

### 1. 技术面分析
- 当前阶段：{technical_result['current_stage']}
- 置信度：{technical_result['confidence']:.1%}
- 各阶段概率：{
    {k: f"{v:.1%}" for k, v in technical_result.get('all_stage_probs', {}).items()}
}
- 特征分析：{technical_result.get('feature_analysis', {})}

"""
        
        # 添加市场面信息
        if market_context:
            prompt += f"""### 2. 市场面分析
- 大盘趋势：{market_context.index_trends}
- 板块趋势：{market_context.sector_trends}
- 相对大盘强度：{market_context.relative_strength_vs_index:+.1%}
- 相对板块强度：{market_context.relative_strength_vs_sector:+.1%}
- 市场情绪：{market_context.market_sentiment}

"""
        
        # 添加信息面信息
        if news_analysis:
            prompt += f"""### 3. 信息面分析
- 新闻情绪：{news_analysis.news_sentiment} (分数: {news_analysis.sentiment_score:+.2f})
- 关键事件：{news_analysis.key_events}
- 政策影响：{news_analysis.policy_impact}
- 风险信号：{news_analysis.risk_signals}
- 机会信号：{news_analysis.opportunity_signals}
- 新闻摘要：{news_analysis.summary}

"""
        
        prompt += """请给出以下分析（用JSON格式返回）：
{
    "stage": "吸筹期/拉升期/派发期/观望期",
    "confidence": 0.0,  // 0-1之间
    "analysis": "综合分析说明，200字以内",
    "recommendation": "具体操作建议，如：建议观望/可轻仓试探/建议减仓等",
    "risk_level": "low/medium/high"
}

判断逻辑：
1. 技术面是主要依据
2. 如果相对大盘/板块强度明显为负，可能降低一个阶段判断
3. 如果新闻情绪极度负面且有重大风险事件，需警惕
4. 如果新闻情绪积极且有机会信号，可适当乐观
"""
        
        return prompt
    
    def _llm_comprehensive_analysis(self, prompt: str) -> Dict:
        """使用LLM进行综合分析"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的股票投资分析师，擅长综合技术面、市场面、信息面进行判断。请用中文回复，只返回JSON格式数据。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        return json.loads(content)
    
    def _rule_comprehensive_analysis(
        self,
        technical_stage: str,
        technical_confidence: float,
        market_context: Optional[MarketContext],
        news_analysis: Optional[NewsAnalysis]
    ) -> Dict:
        """使用规则引擎进行综合分析"""
        
        stage = technical_stage
        confidence = technical_confidence
        adjustments = []
        
        # 市场面调整
        if market_context:
            # 相对强度过低，可能降低判断
            if market_context.relative_strength_vs_index < -0.5:
                adjustments.append("相对大盘明显弱势")
                confidence *= 0.9
            
            if market_context.relative_strength_vs_sector < -0.5:
                adjustments.append("相对板块明显弱势")
                confidence *= 0.9
            
            # 市场情绪极度负面
            if market_context.market_sentiment in ['极度恐慌']:
                adjustments.append("市场情绪极度恐慌")
                confidence *= 0.85
        
        # 信息面调整
        if news_analysis:
            # 新闻情绪极度负面
            if news_analysis.news_sentiment == 'negative' and news_analysis.sentiment_score < -0.5:
                adjustments.append("新闻情绪极度负面")
                confidence *= 0.85
                
                # 如果有重大风险事件，考虑降低阶段
                if len(news_analysis.risk_signals) >= 2:
                    stage = self._downgrade_stage(stage)
                    adjustments.append("存在多重风险信号")
            
            # 新闻情绪积极
            elif news_analysis.news_sentiment == 'positive' and news_analysis.sentiment_score > 0.5:
                adjustments.append("新闻情绪积极")
                confidence = min(confidence * 1.05, 0.99)
        
        # 生成分析文本
        if adjustments:
            analysis = f"技术面判断为{technical_stage}，但考虑{'；'.join(adjustments)}，综合判断为{stage}。"
        else:
            analysis = f"技术面、市场面、信息面综合判断为{stage}。"
        
        # 确定风险等级
        risk_level = self._determine_risk_level(stage, market_context, news_analysis)
        
        return {
            'stage': stage,
            'confidence': confidence,
            'analysis': analysis,
            'recommendation': self._get_recommendation(stage, risk_level, adjustments),
            'risk_level': risk_level
        }
    
    def _downgrade_stage(self, stage: str) -> str:
        """降低阶段判断（更保守）"""
        stage_order = {
            '拉升期': '派发期',
            '派发期': '观望期',
            '观望期': '观望期',
            '吸筹期': '观望期'  # 吸筹期变观望，说明可能还没到底
        }
        return stage_order.get(stage, stage)
    
    def _determine_risk_level(
        self, 
        stage: str, 
        market_context: Optional[MarketContext],
        news_analysis: Optional[NewsAnalysis]
    ) -> str:
        """确定风险等级"""
        risk_score = 0
        
        # 阶段风险
        if stage == '拉升期':
            risk_score += 2  # 高位风险
        elif stage == '派发期':
            risk_score += 3  # 下跌风险
        elif stage == '观望期':
            risk_score += 1
        elif stage == '吸筹期':
            risk_score += 0
        
        # 市场面风险
        if market_context:
            if market_context.market_sentiment in ['极度恐慌']:
                risk_score += 2
            elif market_context.market_sentiment in ['恐慌']:
                risk_score += 1
        
        # 信息面风险
        if news_analysis:
            if news_analysis.news_sentiment == 'negative':
                risk_score += 2
            if len(news_analysis.risk_signals) >= 2:
                risk_score += 1
        
        if risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommendation(self, stage: str, risk_level: str, adjustments: List[str]) -> str:
        """生成操作建议"""
        base_recommendations = {
            '吸筹期': '可逐步建仓，关注底部确认信号',
            '拉升期': '持有为主，可适量加仓',
            '派发期': '建议减仓或离场，避免追高',
            '观望期': '建议观望，等待明确信号'
        }
        
        rec = base_recommendations.get(stage, '建议观望')
        
        # 根据风险等级调整
        if risk_level == 'high':
            rec += '【高风险，谨慎操作】'
        elif risk_level == 'medium':
            rec += '【中等风险】'
        
        return rec
    
    def _get_basic_recommendation(self, stage: str) -> str:
        """获取基础建议"""
        recommendations = {
            '吸筹期': '可逐步建仓',
            '拉升期': '持有或适量加仓',
            '派发期': '建议减仓',
            '观望期': '建议观望'
        }
        return recommendations.get(stage, '建议观望')
