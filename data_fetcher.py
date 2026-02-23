"""
数据获取模块 - Tushare Pro + AKShare + 本地数据库缓存
智能缓存策略：优先从本地获取，不足时补充
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time

from db_manager import StockDatabase

# 延迟导入 tushare 和 akshare（避免启动时网络连接）
ts = None
ak = None

def _import_tushare():
    global ts
    if ts is None:
        import tushare as ts_module
        ts = ts_module
    return ts

def _import_akshare():
    global ak
    if ak is None:
        import akshare as ak_module
        ak = ak_module
    return ak


class DataFetcher:
    """股票数据获取器 - 支持本地缓存"""
    
    def __init__(self, tushare_token: str = ""):
        self.tushare_token = tushare_token
        self.pro = None
        if tushare_token:
            try:
                ts_module = _import_tushare()
                ts_module.set_token(tushare_token)
                self.pro = ts_module.pro_api()
            except:
                pass
        
        # 本地数据库
        self.db = StockDatabase()
    
    def fetch_stock_data(
        self, 
        stock_code: str, 
        days: int = 1000,
        use_cache: bool = True
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """
        获取股票历史数据（智能缓存版本）
        
        Args:
            stock_code: 股票代码
            days: 需要的天数（默认1000天，如果股票上市不足则从上市日开始）
            use_cache: 是否使用本地缓存
        
        Returns:
            (df, source) - source 为 "local", "tushare", "akshare", "mixed" 或 "failed"
        """
        # 标准化股票代码
        stock_code = self._normalize_code(stock_code)
        pure_code = stock_code.split('.')[0]
        
        if use_cache:
            # 1. 先尝试从本地获取
            local_df, available_days = self.db.get_local_data(
                pure_code, 
                min_days=days
            )
            
            # 如果本地数据足够，直接返回（达到目标的50%即可）
            min_required = int(days * 0.5)  # 最少需要目标天数的50%
            if local_df is not None and available_days >= min_required:
                print(f"  📦 本地缓存: {pure_code} ({available_days}天)")
                df = self._process_data(local_df)
                return df, "local"
            
            # 2. 本地数据不足，需要补充
            print(f"  📦 本地数据不足: {pure_code} (有{available_days}天，需要{days}天)")
            
            # 确定需要补充的日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 60)  # 多取一些防止节假日
            
            # 如果本地有数据，从本地最新日期开始补充
            if local_df is not None and not local_df.empty:
                local_end = pd.to_datetime(local_df['date'].max())
                if local_end >= end_date - timedelta(days=5):
                    # 本地数据已经够新，只是总量不够，需要往前补
                    start_date = pd.to_datetime(local_df['date'].min()) - timedelta(days=days - available_days + 30)
                else:
                    # 本地数据较旧，需要更新到最新
                    start_date = local_end - timedelta(days=1)
            
            # 3. 从外部获取补充数据
            new_df = self._fetch_from_external(pure_code, start_date, end_date)
            
            if new_df is not None and not new_df.empty:
                # 4. 保存到本地数据库
                self.db.save_data(pure_code, new_df, source=new_df.attrs.get('source', 'unknown'))
                
                # 5. 合并本地和新增数据
                if local_df is not None:
                    combined = pd.concat([local_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['date'], keep='last')
                    combined = combined.sort_values('date').reset_index(drop=True)
                    
                    # 取最近 days 天
                    if len(combined) > days:
                        combined = combined.tail(days).reset_index(drop=True)
                    
                    df = self._process_data(combined)
                    return df, "mixed"
                else:
                    df = self._process_data(new_df)
                    return df, new_df.attrs.get('source', 'unknown')
        
        # 不使用缓存，直接从外部获取
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60)
        df = self._fetch_from_external(pure_code, start_date, end_date)
        
        if df is not None and not df.empty:
            # 保存到本地
            if use_cache:
                self.db.save_data(pure_code, df, source=df.attrs.get('source', 'unknown'))
            df = self._process_data(df)
            return df, df.attrs.get('source', 'unknown')
        
        return None, "failed"
    
    def _fetch_from_external(
        self, 
        stock_code: str, 
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """从外部数据源获取数据"""
        # 标准化代码
        if '.' not in stock_code:
            if stock_code.startswith('6'):
                ts_code = f"{stock_code}.SH"
            else:
                ts_code = f"{stock_code}.SZ"
        else:
            ts_code = stock_code
            stock_code = stock_code.split('.')[0]
        
        # 1. 尝试 Tushare
        if self.pro:
            try:
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d')
                )
                
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        'trade_date': 'date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'volume',
                        'amount': 'amount'
                    })
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    df.attrs['source'] = 'tushare'
                    return df
                    
            except Exception as e:
                print(f"  ⚠️ Tushare获取失败: {e}")
        
        # 2. 备份使用 AKShare
        try:
            ak_module = _import_akshare()
            df = ak_module.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df.attrs['source'] = 'akshare'
                return df
                
        except Exception as e:
            print(f"  ⚠️ AKShare获取失败: {e}")
        
        return None
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据，计算技术指标"""
        df = df.copy()
        
        # 确保必要列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 涨跌幅
        df['returns'] = df['close'].pct_change()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # 成交量移动平均
        df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        
        # 价格波动率 (20日)
        df['volatility'] = df['returns'].rolling(window=20, min_periods=5).std()
        
        # 相对位置 (当前价格在20日区间的位置)
        high_20 = df['high'].rolling(20, min_periods=5).max()
        low_20 = df['low'].rolling(20, min_periods=5).min()
        df['price_position'] = (df['close'] - low_20) / (high_20 - low_20).replace(0, 1)
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 添加股票代码列
        if 'ts_code' not in df.columns:
            df['ts_code'] = ''
        
        return df
    
    def _normalize_code(self, code: str) -> str:
        """标准化股票代码"""
        code = code.strip()
        if '.' in code:
            return code
        
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.SZ"
        elif code.startswith('8') or code.startswith('4'):
            return f"{code}.BJ"
        return code
    
    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        pure_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
        
        # 先从本地查
        list_date = self.db.get_list_date(pure_code)
        
        try:
            ak_module = _import_akshare()
            df = ak_module.stock_individual_info_em(symbol=pure_code)
            if df is not None and not df.empty:
                name = df[df['item'] == '股票简称']['value'].values[0]
                # 保存到本地
                list_date_val = df[df['item'] == '上市日期']['value'].values
                if len(list_date_val) > 0:
                    self.db.save_stock_info(
                        pure_code, 
                        stock_name=name,
                        list_date=str(list_date_val[0])
                    )
                return name
        except:
            pass
        
        return stock_code
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return self.db.get_cache_stats()
    
    def clear_cache(self, stock_code: str = None):
        """清除缓存"""
        if stock_code:
            pure_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            self.db.delete_stock_data(pure_code)
            print(f"已清除 {pure_code} 的缓存")
        else:
            # 清除所有缓存（删除数据库文件）
            import os
            db_path = self.db.db_path
            if db_path.exists():
                os.remove(db_path)
                print("已清除所有缓存")
