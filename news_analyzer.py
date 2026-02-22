"""
信息面分析模块 - 新闻、公告、市场情绪
支持多数据源：Tavily搜索、Tushare新闻、AKShare财经新闻
"""
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class NewsAnalysis:
    """新闻分析结果"""
    stock_code: str
    stock_name: str
    
    # 新闻摘要
    recent_news: List[Dict]  # 近期相关新闻列表
    news_sentiment: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1 ~ 1
    
    # 关键事件
    key_events: List[str]  # 关键事件摘要
    
    # 行业动态
    industry_news: List[Dict]  # 行业相关新闻
    policy_impact: str  # 政策影响评估
    
    # 风险信号
    risk_signals: List[str]  # 风险警示
    opportunity_signals: List[str]  # 机会信号
    
    # 综合分析
    summary: str  # LLM综合摘要
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            '股票代码': self.stock_code,
            '股票名称': self.stock_name,
            '新闻情绪': self.news_sentiment,
            '情绪分数': f"{self.sentiment_score:+.2f}",
            '近期要闻': self.recent_news[:3] if self.recent_news else [],
            '关键事件': self.key_events,
            '行业动态': self.industry_news[:2] if self.industry_news else [],
            '政策影响': self.policy_impact,
            '风险信号': self.risk_signals,
            '机会信号': self.opportunity_signals,
            '综合摘要': self.summary,
        }


class NewsAnalyzer:
    """新闻分析器 - 使用Tavily免费搜索API"""
    
    def __init__(
        self, 
        tavily_api_key: Optional[str] = None,
        tushare_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: Optional[str] = None
    ):
        # Tavily API Key（免费版）
        self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY', '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')
        
        self.tushare_token = tushare_token
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url or "https://api.openai.com/v1"
        self.openai_model = openai_model or "gpt-3.5-turbo"
        
        # 初始化Tushare
        self.pro = None
        if self.tushare_token:
            try:
                import tushare as ts
                ts.set_token(self.tushare_token)
                self.pro = ts.pro_api()
            except:
                pass
    
    def analyze_stock_news(
        self, 
        stock_code: str, 
        stock_name: str,
        industry: str = "",
        days: int = 7
    ) -> NewsAnalysis:
        """
        分析股票相关新闻
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            industry: 所属行业
            days: 分析近N天的新闻
        
        Returns:
            NewsAnalysis 对象
        """
        # 1. 使用Tavily搜索新闻
        search_results = self._tavily_search(stock_code, stock_name, industry)
        
        # 2. 获取本地新闻数据
        local_news = self._fetch_local_news(stock_code, industry, days)
        
        # 3. 合并新闻数据
        all_news = self._merge_news(search_results, local_news)
        
        # 4. 使用LLM分析新闻
        analysis_result = self._llm_analyze_news(
            stock_code, stock_name, industry, all_news
        )
        
        # 5. 构建结果
        return NewsAnalysis(
            stock_code=stock_code,
            stock_name=stock_name,
            recent_news=all_news.get('stock_news', []),
            news_sentiment=analysis_result.get('sentiment', 'neutral'),
            sentiment_score=analysis_result.get('sentiment_score', 0.0),
            key_events=analysis_result.get('key_events', []),
            industry_news=all_news.get('industry_news', []),
            policy_impact=analysis_result.get('policy_impact', '暂无重大影响'),
            risk_signals=analysis_result.get('risk_signals', []),
            opportunity_signals=analysis_result.get('opportunity_signals', []),
            summary=analysis_result.get('summary', '')
        )
    
    def _tavily_search(self, stock_code: str, stock_name: str, industry: str) -> Dict:
        """
        使用Tavily API搜索新闻
        Tavily提供免费搜索，每月1000次调用
        """
        news_data = {
            'stock_news': [],
            'industry_news': [],
            'search_results': []
        }
        
        try:
            url = "https://api.tavily.com/search"
            
            # 搜索股票相关新闻
            queries = [
                f"{stock_name} {stock_code} 股票 新闻 公告 最近一周",
                f"{stock_name} 业绩 财报 研报",
                f"{industry} 行业 政策 新闻" if industry else f"{stock_name} 行业动态"
            ]
            
            for query in queries:
                try:
                    payload = {
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": True,
                        "max_results": 5,
                        "time_range": "week"  # 最近一周
                    }
                    
                    response = requests.post(url, json=payload, timeout=15)
                    response.raise_for_status()
                    result = response.json()
                    
                    # 解析搜索结果
                    if 'results' in result:
                        for item in result['results']:
                            news_item = {
                                'title': item.get('title', ''),
                                'content': item.get('content', ''),
                                'url': item.get('url', ''),
                                'source': item.get('source', ''),
                                'published_date': item.get('published_date', ''),
                            }
                            news_data['search_results'].append(news_item)
                            
                            # 分类新闻
                            if stock_name in query or stock_code in query:
                                news_data['stock_news'].append(news_item)
                            else:
                                news_data['industry_news'].append(news_item)
                    
                    # Tavily提供的智能摘要
                    if 'answer' in result and result['answer']:
                        news_data['tavily_summary'] = result['answer']
                        
                except Exception as e:
                    print(f"  ⚠️ Tavily搜索失败 '{query}': {e}")
                    continue
            
        except Exception as e:
            print(f"  ⚠️ Tavily API调用失败: {e}")
        
        return news_data
    
    def _fetch_local_news(self, stock_code: str, industry: str, days: int) -> Dict:
        """获取本地新闻数据（Tushare/AKShare）"""
        news_data = {
            'stock_news': [],
            'industry_news': []
        }
        
        # 1. 尝试从Tushare获取新闻
        if self.pro:
            try:
                df_news = self.pro.news(
                    ts_code=stock_code,
                    start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d')
                )
                if df_news is not None and not df_news.empty:
                    news_data['stock_news'].extend(df_news.to_dict('records'))
            except Exception as e:
                pass  # 静默失败
        
        # 2. 尝试从AKShare获取新闻
        try:
            import akshare as ak
            pure_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            try:
                ak_news = ak.stock_news_em(symbol=pure_code)
                if ak_news is not None and not ak_news.empty:
                    news_data['stock_news'].extend(ak_news.head(10).to_dict('records'))
            except:
                pass
            
            if industry:
                try:
                    industry_news = ak.stock_sector_news_em(sector=industry)
                    if industry_news is not None and not industry_news.empty:
                        news_data['industry_news'] = industry_news.head(5).to_dict('records')
                except:
                    pass
        except:
            pass
        
        return news_data
    
    def _merge_news(self, search_results: Dict, local_news: Dict) -> Dict:
        """合并搜索结果和本地新闻"""
        merged = {
            'stock_news': [],
            'industry_news': [],
            'tavily_summary': search_results.get('tavily_summary', '')
        }
        
        # 去重合并
        seen_urls = set()
        
        for news in search_results.get('stock_news', []) + local_news.get('stock_news', []):
            url = news.get('url', '') or str(news)
            if url not in seen_urls:
                merged['stock_news'].append(news)
                seen_urls.add(url)
        
        for news in search_results.get('industry_news', []) + local_news.get('industry_news', []):
            url = news.get('url', '') or str(news)
            if url not in seen_urls:
                merged['industry_news'].append(news)
                seen_urls.add(url)
        
        return merged
    
    def _llm_analyze_news(
        self, 
        stock_code: str, 
        stock_name: str,
        industry: str,
        news_data: Dict
    ) -> Dict:
        """使用LLM分析新闻"""
        
        # 优先使用OpenAI兼容API
        if self.openai_api_key:
            return self._openai_analysis(stock_code, stock_name, industry, news_data)
        
        # 无API时使用Tavily摘要或基础分析
        if news_data.get('tavily_summary'):
            return self._tavily_summary_analysis(news_data['tavily_summary'])
        
        return self._basic_analysis(news_data)
    
    def _openai_analysis(
        self, 
        stock_code: str, 
        stock_name: str,
        industry: str,
        news_data: Dict
    ) -> Dict:
        """使用OpenAI兼容API分析新闻"""
        try:
            # 构建新闻文本
            news_text = ""
            
            # 添加Tavily摘要
            if news_data.get('tavily_summary'):
                news_text += f"【搜索摘要】\n{news_data['tavily_summary']}\n\n"
            
            # 添加具体新闻
            news_text += "【相关新闻】\n"
            for i, news in enumerate(news_data.get('stock_news', [])[:8], 1):
                if isinstance(news, dict):
                    title = news.get('title', '')
                    content = news.get('content', news.get('摘要', ''))
                    if title or content:
                        news_text += f"{i}. {title}\n{content}\n\n"
            
            # 添加行业新闻
            if news_data.get('industry_news'):
                news_text += "【行业动态】\n"
                for i, news in enumerate(news_data['industry_news'][:3], 1):
                    if isinstance(news, dict):
                        title = news.get('title', '')
                        if title:
                            news_text += f"{i}. {title}\n"
            
            prompt = f"""请分析以下关于 {stock_name}({stock_code}) {industry} 的新闻信息：

{news_text}

请提供以下分析（用JSON格式返回）：
{{
    "sentiment": "positive/negative/neutral",
    "sentiment_score": 0.0,  // -1到1之间，正值为利好，负值为利空
    "key_events": ["事件1", "事件2", "事件3"],  // 最近重要事件，最多5条
    "policy_impact": "政策影响描述，如果没有则写'暂无重大影响'",
    "risk_signals": ["风险1", "风险2"],  // 风险信号，最多3条，没有则留空数组
    "opportunity_signals": ["机会1", "机会2"],  // 机会信号，最多3条，没有则留空数组
    "summary": "综合摘要，100字以内，总结对投资的影响"
}}

注意：
1. sentiment_score 范围是 -1.0 到 1.0
2. 如果有业绩预增、订单增长、政策利好等，sentiment为positive
3. 如果有业绩预减、违规调查、大股东减持等，sentiment为negative
4. 关键事件要具体，包含事件类型和影响"""

            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的股票分析师，擅长从新闻中提取关键信息并判断市场情绪。请用中文回复，只返回JSON格式数据。"
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
            
        except Exception as e:
            print(f"  ⚠️ OpenAI分析失败: {e}")
            if news_data.get('tavily_summary'):
                return self._tavily_summary_analysis(news_data['tavily_summary'])
            return self._basic_analysis(news_data)
    
    def _tavily_summary_analysis(self, summary: str) -> Dict:
        """基于Tavily摘要的基础分析"""
        # 简单的关键词匹配
        positive_words = ['利好', '增长', '盈利', '突破', '合作', '订单', '增持', '回购', '上涨', '预增', '超预期']
        negative_words = ['利空', '亏损', '下降', '违规', '调查', '减持', '债务', '诉讼', '下跌', '预减', '不及预期']
        
        summary_lower = summary.lower()
        
        pos_count = sum(1 for w in positive_words if w in summary_lower)
        neg_count = sum(1 for w in negative_words if w in summary_lower)
        
        total = pos_count + neg_count
        if total == 0:
            sentiment = 'neutral'
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2:
                sentiment = 'positive'
            elif score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'sentiment_score': score,
            'key_events': ['基于搜索摘要分析'],
            'policy_impact': '详见搜索摘要',
            'risk_signals': [w for w in negative_words if w in summary_lower][:3],
            'opportunity_signals': [w for w in positive_words if w in summary_lower][:3],
            'summary': summary[:200] + '...' if len(summary) > 200 else summary
        }
    
    def _basic_analysis(self, news_data: Dict) -> Dict:
        """基础分析（无API时使用）"""
        positive_words = ['利好', '增长', '盈利', '突破', '合作', '订单', '增持', '回购', '上涨', '预增']
        negative_words = ['利空', '亏损', '下降', '违规', '调查', '减持', '债务', '诉讼', '下跌', '预减']
        
        news_text = ""
        for news in news_data.get('stock_news', []):
            if isinstance(news, dict):
                news_text += str(news.get('title', '')) + str(news.get('content', ''))
        
        pos_count = sum(1 for w in positive_words if w in news_text)
        neg_count = sum(1 for w in negative_words if w in news_text)
        
        total = pos_count + neg_count
        if total == 0:
            sentiment = 'neutral'
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2:
                sentiment = 'positive'
            elif score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'sentiment_score': score,
            'key_events': [],
            'policy_impact': '暂无分析',
            'risk_signals': [],
            'opportunity_signals': [],
            'summary': f'基于关键词分析，新闻情绪为{sentiment}'
        }
    
    def get_hot_stocks_analysis(self) -> Dict:
        """获取市场热股分析"""
        try:
            import akshare as ak
            
            # 获取东方财富热股榜
            hot_stocks = ak.stock_hot_rank_em()
            
            # 获取涨停股
            try:
                limit_up = ak.stock_zt_pool_em(date=datetime.now().strftime('%Y%m%d'))
            except:
                limit_up = pd.DataFrame()
            
            return {
                '热门股票': hot_stocks.head(10).to_dict('records') if not hot_stocks.empty else [],
                '涨停股票': limit_up.to_dict('records') if not limit_up.empty else [],
                '分析时间': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
        except Exception as e:
            print(f"  ⚠️ 获取热股分析失败: {e}")
            return {}


class AnnouncementMonitor:
    """公告监控器 - 监控重要公告"""
    
    def __init__(self, tushare_token: Optional[str] = None):
        self.tushare_token = tushare_token
        self.pro = None
        if tushare_token:
            try:
                import tushare as ts
                ts.set_token(tushare_token)
                self.pro = ts.pro_api()
            except:
                pass
    
    def get_recent_announcements(self, stock_code: str, days: int = 7) -> List[Dict]:
        """获取近期公告"""
        announcements = []
        
        if not self.pro:
            return announcements
        
        try:
            df = self.pro.anns(
                ts_code=stock_code,
                start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            
            if df is not None and not df.empty:
                announcements = df.to_dict('records')
        except Exception as e:
            print(f"  ⚠️ 获取公告失败: {e}")
        
        return announcements
    
    def classify_announcement(self, title: str) -> str:
        """分类公告类型并评估重要性"""
        important_keywords = [
            '业绩预告', '业绩快报', '年报', '半年报', '季报',
            '重大资产重组', '收购', '并购', '借壳',
            '定增', '增发', '配股',
            '股权激励', '回购',
            '股东减持', '股东增持',
            '停牌', '复牌',
            '退市', 'ST',
            '立案调查', '处罚',
            '高管变动', '董事长', '总经理'
        ]
        
        title_lower = title.lower()
        
        for keyword in important_keywords:
            if keyword in title_lower:
                return '重要'
        
        return '一般'
