"""
市场上下文获取模块 - 大盘、板块、资金流向
用于对比个股与市场的相对表现
"""
import pandas as pd
import numpy as np
import tushare as ts
import akshare as ak
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class MarketContext:
    """市场上下文数据"""
    # 大盘数据
    index_data: Dict[str, pd.DataFrame]  # 各指数数据
    index_trends: Dict[str, str]  # 大盘趋势: 'up', 'down', 'sideways'
    
    # 板块数据
    sector_data: Dict[str, pd.DataFrame]  # 所属板块数据
    sector_trends: Dict[str, str]  # 板块趋势
    sector_ranks: Dict[str, int]  # 板块排名
    
    # 相对强度
    relative_strength_vs_index: float  # 相对大盘强度 (-1 ~ 1)
    relative_strength_vs_sector: float  # 相对板块强度 (-1 ~ 1)
    
    # 资金流向
    money_flow: Optional[pd.DataFrame]  # 个股资金流向
    north_bound: Optional[pd.DataFrame]  # 北向资金
    
    # 市场情绪
    market_sentiment: str  # '极度恐慌', '恐慌', '中性', '乐观', '极度乐观'
    
    def to_dict(self) -> dict:
        """转换为字典（用于LLM输入）"""
        return {
            '大盘趋势': self.index_trends,
            '板块趋势': self.sector_trends,
            '板块排名': self.sector_ranks,
            '相对大盘强度': f"{self.relative_strength_vs_index:+.1%}",
            '相对板块强度': f"{self.relative_strength_vs_sector:+.1%}",
            '市场情绪': self.market_sentiment,
        }


class MarketContextFetcher:
    """市场上下文获取器"""
    
    # 主要指数代码
    INDEX_CODES = {
        '上证指数': '000001.SH',
        '深证成指': '399001.SZ',
        '创业板指': '399006.SZ',
        '沪深300': '000300.SH',
        '中证500': '000905.SH',
        '科创50': '000688.SH',
    }
    
    def __init__(self, tushare_token: str = ""):
        self.tushare_token = tushare_token
        self.pro = None
        if tushare_token:
            try:
                ts.set_token(tushare_token)
                self.pro = ts.pro_api()
            except:
                pass
    
    def get_market_context(
        self, 
        stock_code: str, 
        stock_df: pd.DataFrame,
        days: int = 60
    ) -> MarketContext:
        """
        获取股票的市场上下文
        
        Args:
            stock_code: 股票代码
            stock_df: 个股历史数据
            days: 分析天数
        
        Returns:
            MarketContext 对象
        """
        # 1. 获取大盘数据
        index_data = self._fetch_index_data(days)
        index_trends = self._analyze_trends(index_data)
        
        # 2. 获取板块数据
        sector_data, sector_codes = self._fetch_sector_data(stock_code, days)
        sector_trends = self._analyze_trends(sector_data)
        sector_ranks = self._get_sector_rankings(sector_codes)
        
        # 3. 计算相对强度
        rs_index = self._calculate_relative_strength(stock_df, index_data.get('沪深300'))
        rs_sector = self._calculate_relative_strength(stock_df, list(sector_data.values())[0] if sector_data else None)
        
        # 4. 获取资金流向
        money_flow = self._fetch_money_flow(stock_code)
        north_bound = self._fetch_north_bound_flow()
        
        # 5. 判断市场情绪
        sentiment = self._analyze_market_sentiment(index_data, north_bound)
        
        return MarketContext(
            index_data=index_data,
            index_trends=index_trends,
            sector_data=sector_data,
            sector_trends=sector_trends,
            sector_ranks=sector_ranks,
            relative_strength_vs_index=rs_index,
            relative_strength_vs_sector=rs_sector,
            money_flow=money_flow,
            north_bound=north_bound,
            market_sentiment=sentiment
        )
    
    def _fetch_index_data(self, days: int) -> Dict[str, pd.DataFrame]:
        """获取主要指数数据"""
        index_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        for name, code in self.INDEX_CODES.items():
            try:
                if self.pro:
                    df = self.pro.index_daily(
                        ts_code=code,
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d')
                    )
                    if df is not None and not df.empty:
                        df = df.rename(columns={
                            'trade_date': 'date',
                            'close': 'close',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'vol': 'volume',
                            'pct_chg': 'returns'
                        })
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)
                        # 计算涨跌幅（如果pct_chg不存在）
                        if 'returns' not in df.columns:
                            df['returns'] = df['close'].pct_change() * 100
                        index_data[name] = df
            except Exception as e:
                print(f"  ⚠️ 获取{name}数据失败: {e}")
        
        return index_data
    
    def _fetch_sector_data(self, stock_code: str, days: int) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """获取股票所属板块数据"""
        sector_data = {}
        sector_codes = []
        
        # 标准化代码
        pure_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
        
        try:
            # 获取股票所属行业
            if self.pro:
                # 使用Tushare获取行业信息
                df_basic = self.pro.stock_basic(ts_code=stock_code, fields='ts_code,industry')
                if df_basic is not None and not df_basic.empty:
                    industry = df_basic['industry'].values[0]
                    if industry:
                        # 获取行业指数数据（这里简化处理，实际可能需要映射到行业指数代码）
                        pass
            
            # 使用AKShare获取板块数据
            # 获取个股所属概念板块
            try:
                concept_df = ak.stock_board_concept_name_em()
                # 获取前5个热门概念板块的数据
                for _, row in concept_df.head(5).iterrows():
                    sector_name = row['板块名称']
                    try:
                        sector_hist = ak.stock_board_concept_hist_em(
                            symbol=sector_name, 
                            period="daily",
                            adjust="qfq"
                        )
                        if sector_hist is not None and not sector_hist.empty:
                            sector_hist = sector_hist.rename(columns={
                                '日期': 'date',
                                '收盘': 'close',
                                '开盘': 'open',
                                '最高': 'high',
                                '最低': 'low',
                                '成交量': 'volume',
                                '涨跌幅': 'returns'
                            })
                            sector_hist['date'] = pd.to_datetime(sector_hist['date'])
                            sector_hist = sector_hist.sort_values('date').reset_index(drop=True)
                            sector_data[sector_name] = sector_hist
                            sector_codes.append(sector_name)
                    except:
                        continue
            except Exception as e:
                print(f"  ⚠️ 获取板块数据失败: {e}")
                
        except Exception as e:
            print(f"  ⚠️ 获取行业信息失败: {e}")
        
        return sector_data, sector_codes
    
    def _analyze_trends(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """分析趋势"""
        trends = {}
        for name, df in data_dict.items():
            if df is None or len(df) < 20:
                trends[name] = 'unknown'
                continue
            
            # 计算短期和中期均线
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            
            latest = df.iloc[-1]
            
            # 判断趋势
            if latest['ma5'] > latest['ma20'] * 1.02:
                trends[name] = '上升趋势'
            elif latest['ma5'] < latest['ma20'] * 0.98:
                trends[name] = '下降趋势'
            else:
                trends[name] = '横盘震荡'
        
        return trends
    
    def _get_sector_rankings(self, sector_codes: List[str]) -> Dict[str, int]:
        """获取板块排名（按近期涨幅）"""
        rankings = {}
        try:
            # 获取板块排名数据
            sector_spot = ak.stock_board_concept_name_em()
            for i, code in enumerate(sector_codes):
                # 查找板块排名
                match = sector_spot[sector_spot['板块名称'] == code]
                if not match.empty:
                    rankings[code] = int(match.index[0]) + 1
                else:
                    rankings[code] = 999
        except:
            for code in sector_codes:
                rankings[code] = 999
        
        return rankings
    
    def _calculate_relative_strength(
        self, 
        stock_df: pd.DataFrame, 
        benchmark_df: Optional[pd.DataFrame]
    ) -> float:
        """
        计算相对强度 (RS)
        RS = (个股收益率 - 基准收益率) / 基准波动率
        返回 -1 ~ 1 的标准化值
        """
        if benchmark_df is None or stock_df is None:
            return 0.0
        
        try:
            # 对齐日期
            stock_df = stock_df.copy()
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            
            benchmark_df = benchmark_df.copy()
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            
            # 合并数据
            merged = pd.merge(
                stock_df[['date', 'close']], 
                benchmark_df[['date', 'close']], 
                on='date', 
                suffixes=('_stock', '_bench')
            )
            
            if len(merged) < 20:
                return 0.0
            
            # 计算收益率
            merged['stock_ret'] = merged['close_stock'].pct_change()
            merged['bench_ret'] = merged['close_bench'].pct_change()
            
            # 计算相对强度（近20日）
            recent = merged.tail(20)
            stock_cumret = (1 + recent['stock_ret']).prod() - 1
            bench_cumret = (1 + recent['bench_ret']).prod() - 1
            bench_volatility = recent['bench_ret'].std()
            
            if bench_volatility == 0:
                return 0.0
            
            # 标准化到 -1 ~ 1
            rs = (stock_cumret - bench_cumret) / (bench_volatility + 0.01)
            rs = np.clip(rs, -1, 1)
            
            return float(rs)
            
        except Exception as e:
            print(f"  ⚠️ 计算相对强度失败: {e}")
            return 0.0
    
    def _fetch_money_flow(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取个股资金流向"""
        try:
            pure_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            df = ak.stock_individual_fund_flow(stock=pure_code, market="sh" if pure_code.startswith('6') else "sz")
            return df
        except Exception as e:
            print(f"  ⚠️ 获取资金流向失败: {e}")
            return None
    
    def _fetch_north_bound_flow(self) -> Optional[pd.DataFrame]:
        """获取北向资金流向"""
        try:
            df = ak.stock_hsgt_hist_em(symbol="北上资金")
            return df
        except Exception as e:
            print(f"  ⚠️ 获取北向资金失败: {e}")
            return None
    
    def _analyze_market_sentiment(
        self, 
        index_data: Dict[str, pd.DataFrame],
        north_bound: Optional[pd.DataFrame]
    ) -> str:
        """分析市场情绪"""
        try:
            # 基于大盘近期表现判断
            scores = []
            
            for name, df in index_data.items():
                if df is None or len(df) < 5:
                    continue
                
                # 近5日涨跌幅
                recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                scores.append(recent_return)
            
            if not scores:
                return '中性'
            
            avg_score = np.mean(scores)
            
            # 结合北向资金
            if north_bound is not None and not north_bound.empty:
                recent_north = north_bound.tail(5)['净流入'].sum() if '净流入' in north_bound.columns else 0
                if recent_north > 100:  # 100亿
                    avg_score += 1
                elif recent_north < -100:
                    avg_score -= 1
            
            # 映射到情绪
            if avg_score > 3:
                return '极度乐观'
            elif avg_score > 1:
                return '乐观'
            elif avg_score < -3:
                return '极度恐慌'
            elif avg_score < -1:
                return '恐慌'
            else:
                return '中性'
                
        except:
            return '中性'
    
    def get_sector_rotation_analysis(self) -> Dict:
        """
        获取板块轮动分析
        返回近期强势的板块列表
        """
        try:
            # 获取板块涨幅排名
            sector_spot = ak.stock_board_concept_name_em()
            
            # 取涨幅前10的板块
            top_sectors = sector_spot.head(10)[['板块名称', '涨跌幅', '换手率', '上涨家数', '下跌家数']]
            
            return {
                '强势板块': top_sectors.to_dict('records'),
                '分析日期': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            print(f"  ⚠️ 获取板块轮动分析失败: {e}")
            return {}
