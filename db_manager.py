"""
股票数据本地数据库管理模块
使用 SQLite 存储历史数据，减少外部 API 查询
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import threading
import json

from config import DATA_DIR


class StockDatabase:
    """股票数据本地数据库"""
    
    def __init__(self):
        self.db_path = DATA_DIR / "stock_data.db"
        self._lock = threading.Lock()
        self._init_db()
        self._migrate_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 股票历史数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_daily (
                    stock_code TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    source TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (stock_code, trade_date)
                )
            """)
            
            # 股票基本信息表（记录上市日期等）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    stock_code TEXT PRIMARY KEY,
                    stock_name TEXT,
                    list_date DATE,
                    total_shares REAL,
                    industry TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 数据更新记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS update_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    start_date DATE,
                    end_date DATE,
                    records_count INTEGER,
                    source TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 周期预测表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycle_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    analysis_date DATE NOT NULL,
                    current_stage TEXT,
                    cycle_start_date DATE,
                    cycle_estimated_end DATE,
                    cycle_progress REAL,
                    days_in_cycle INTEGER,
                    estimated_total_days INTEGER,
                    hist_avg_duration INTEGER,
                    hist_min_duration INTEGER,
                    hist_max_duration INTEGER,
                    current_price REAL,
                    price_target_low REAL,
                    price_target_high REAL,
                    price_target_mean REAL,
                    pred_5d_stage TEXT,
                    pred_5d_price_low REAL,
                    pred_5d_price_high REAL,
                    pred_5d_confidence REAL,
                    pred_20d_stage TEXT,
                    pred_20d_price_low REAL,
                    pred_20d_price_high REAL,
                    pred_20d_confidence REAL,
                    reasoning TEXT,
                    key_indicators TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 预测准确性评估表（用于反馈优化）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    stock_code TEXT NOT NULL,
                    analysis_date DATE,
                    prediction_type TEXT,
                    predicted_stage TEXT,
                    predicted_price_low REAL,
                    predicted_price_high REAL,
                    actual_date DATE,
                    actual_price REAL,
                    actual_stage TEXT,
                    price_error_rate REAL,
                    stage_correct INTEGER,
                    accuracy_score REAL,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES cycle_predictions(id)
                )
            """)
            
            # 创建索引优化查询
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_daily_code_date 
                ON stock_daily(stock_code, trade_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_code_date 
                ON cycle_predictions(stock_code, analysis_date)
            """)
            
            conn.commit()
    
    def _migrate_db(self):
        """数据库迁移 - 添加新列"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 检查 cycle_predictions 表是否需要添加新列
            cursor.execute("PRAGMA table_info(cycle_predictions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # 添加历史统计相关列（如果不存在）
            new_columns = {
                'hist_avg_duration': 'INTEGER',
                'hist_min_duration': 'INTEGER',
                'hist_max_duration': 'INTEGER'
            }
            
            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    try:
                        cursor.execute(f"ALTER TABLE cycle_predictions ADD COLUMN {col_name} {col_type}")
                        print(f"✅ 添加列 {col_name} 到 cycle_predictions 表")
                    except Exception as e:
                        print(f"⚠️ 添加列 {col_name} 失败: {e}")
            
            conn.commit()
    
    def get_local_data(
        self, 
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_days: int = 1000
    ) -> Tuple[Optional[pd.DataFrame], int]:
        """
        从本地数据库获取数据
        
        Returns:
            (df, available_days) - df 可能为 None，available_days 表示本地有多少天数据
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM stock_daily WHERE stock_code = ?"
            params = [stock_code]
            
            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY trade_date ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return None, 0
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.rename(columns={'trade_date': 'date'})
            
            available_days = len(df)
            
            # 如果数据足够，直接返回
            min_required = int(min_days * 0.5)  # 最少需要目标天数的50%
            if available_days >= min_required:
                return df, available_days
            
            return df, available_days
    
    def get_date_range(self, stock_code: str) -> Tuple[Optional[str], Optional[str]]:
        """获取本地数据的日期范围"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MIN(trade_date), MAX(trade_date) FROM stock_daily WHERE stock_code = ?",
                (stock_code,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return result[0], result[1]
            return None, None
    
    def save_data(
        self, 
        stock_code: str, 
        df: pd.DataFrame,
        source: str = 'unknown'
    ) -> int:
        """
        保存数据到本地数据库
        
        Returns:
            保存的记录数
        """
        if df.empty:
            return 0
        
        # 准备数据
        save_df = df.copy()
        save_df['stock_code'] = stock_code
        save_df['source'] = source
        
        # 统一列名
        if 'date' in save_df.columns:
            save_df = save_df.rename(columns={'date': 'trade_date'})
        
        # 确保 trade_date 是字符串格式
        save_df['trade_date'] = pd.to_datetime(save_df['trade_date']).dt.strftime('%Y-%m-%d')
        
        # 选择需要的列
        columns = ['stock_code', 'trade_date', 'open', 'high', 'low', 'close', 
                   'volume', 'amount', 'source']
        save_df = save_df[[col for col in columns if col in save_df.columns]]
        
        # 使用 REPLACE 语句处理重复数据
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            records = save_df.to_dict('records')
            inserted = 0
            
            for record in records:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO stock_daily 
                        (stock_code, trade_date, open, high, low, close, volume, amount, source, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        record.get('stock_code'),
                        record.get('trade_date'),
                        record.get('open'),
                        record.get('high'),
                        record.get('low'),
                        record.get('close'),
                        record.get('volume'),
                        record.get('amount'),
                        record.get('source', 'unknown')
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"保存单条记录失败: {e}")
                    continue
            
            conn.commit()
            
            # 记录更新日志
            if inserted > 0:
                cursor.execute("""
                    INSERT INTO update_log (stock_code, start_date, end_date, records_count, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    stock_code,
                    save_df['trade_date'].min(),
                    save_df['trade_date'].max(),
                    inserted,
                    source
                ))
                conn.commit()
            
            return inserted
    
    def save_prediction(self, prediction) -> int:
        """
        保存周期预测到数据库
        
        Returns:
            预测记录ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO cycle_predictions (
                    stock_code, analysis_date, current_stage,
                    cycle_start_date, cycle_estimated_end, cycle_progress,
                    days_in_cycle, estimated_total_days,
                    hist_avg_duration, hist_min_duration, hist_max_duration, hist_sample_count,
                    current_price, price_target_low, price_target_high, price_target_mean,
                    pred_5d_stage, pred_5d_price_low, pred_5d_price_high, pred_5d_confidence,
                    pred_20d_stage, pred_20d_price_low, pred_20d_price_high, pred_20d_confidence,
                    reasoning, key_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.stock_code,
                prediction.analysis_date,
                prediction.current_stage,
                prediction.cycle_start_date,
                prediction.cycle_estimated_end,
                prediction.cycle_progress,
                prediction.days_in_cycle,
                prediction.estimated_total_days,
                prediction.hist_avg_duration,
                prediction.hist_min_duration,
                prediction.hist_max_duration,
                prediction.hist_sample_count,
                prediction.current_price,
                prediction.price_target_low,
                prediction.price_target_high,
                prediction.price_target_mean,
                prediction.pred_5d_stage,
                prediction.pred_5d_price_low,
                prediction.pred_5d_price_high,
                prediction.pred_5d_confidence,
                prediction.pred_20d_stage,
                prediction.pred_20d_price_low,
                prediction.pred_20d_price_high,
                prediction.pred_20d_confidence,
                prediction.reasoning,
                json.dumps(prediction.key_indicators, ensure_ascii=False)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_predictions(self, stock_code: str, limit: int = 10) -> List[dict]:
        """获取某只股票的历史预测"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM cycle_predictions 
                WHERE stock_code = ? 
                ORDER BY analysis_date DESC 
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(stock_code, limit))
            
            if df.empty:
                return []
            
            # 解析 JSON
            if 'key_indicators' in df.columns:
                df['key_indicators'] = df['key_indicators'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df.to_dict('records')
    
    def evaluate_prediction(
        self,
        prediction_id: int,
        stock_code: str,
        prediction_type: str,
        actual_price: float,
        actual_stage: str,
        analysis_date: str
    ) -> dict:
        """
        评估预测准确性
        
        Returns:
            评估结果
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取原始预测
            cursor.execute(
                "SELECT * FROM cycle_predictions WHERE id = ?",
                (prediction_id,)
            )
            pred = cursor.fetchone()
            
            if not pred:
                return {'error': '预测记录不存在'}
            
            # 解析预测数据
            columns = [desc[0] for desc in cursor.description]
            pred_dict = dict(zip(columns, pred))
            
            # 计算准确性
            if prediction_type == '5d':
                pred_low = pred_dict['pred_5d_price_low']
                pred_high = pred_dict['pred_5d_price_high']
                pred_stage = pred_dict['pred_5d_stage']
            elif prediction_type == '20d':
                pred_low = pred_dict['pred_20d_price_low']
                pred_high = pred_dict['pred_20d_price_high']
                pred_stage = pred_dict['pred_20d_stage']
            else:
                pred_low = pred_dict['price_target_low']
                pred_high = pred_dict['price_target_high']
                pred_stage = pred_dict['current_stage']
            
            # 价格误差率
            pred_mean = (pred_low + pred_high) / 2
            price_error_rate = abs(actual_price - pred_mean) / pred_mean
            
            # 阶段预测是否正确
            stage_correct = 1 if pred_stage == actual_stage else 0
            
            # 综合评分 (价格权重60%，阶段权重40%)
            price_score = max(0, 1 - price_error_rate)
            accuracy_score = price_score * 0.6 + stage_correct * 0.4
            
            # 保存评估结果
            cursor.execute("""
                INSERT INTO prediction_accuracy (
                    prediction_id, stock_code, analysis_date, prediction_type,
                    predicted_stage, predicted_price_low, predicted_price_high,
                    actual_date, actual_price, actual_stage,
                    price_error_rate, stage_correct, accuracy_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, DATE('now'), ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                stock_code,
                analysis_date,
                prediction_type,
                pred_stage,
                pred_low,
                pred_high,
                actual_price,
                actual_stage,
                price_error_rate,
                stage_correct,
                accuracy_score
            ))
            
            conn.commit()
            
            return {
                'prediction_id': prediction_id,
                'price_error_rate': price_error_rate,
                'stage_correct': stage_correct,
                'accuracy_score': accuracy_score,
                'predicted_stage': pred_stage,
                'actual_stage': actual_stage
            }
    
    def get_prediction_accuracy_stats(self, stock_code: str = None) -> dict:
        """获取预测准确性统计"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if stock_code:
                cursor.execute("""
                    SELECT AVG(accuracy_score), AVG(price_error_rate), 
                           AVG(stage_correct), COUNT(*)
                    FROM prediction_accuracy
                    WHERE stock_code = ?
                """, (stock_code,))
            else:
                cursor.execute("""
                    SELECT AVG(accuracy_score), AVG(price_error_rate), 
                           AVG(stage_correct), COUNT(*)
                    FROM prediction_accuracy
                """)
            
            result = cursor.fetchone()
            
            return {
                'avg_accuracy_score': round(result[0], 4) if result[0] else 0,
                'avg_price_error_rate': round(result[1], 4) if result[1] else 0,
                'stage_correct_rate': round(result[2], 4) if result[2] else 0,
                'total_evaluations': result[3]
            }
    
    def get_list_date(self, stock_code: str) -> Optional[str]:
        """获取股票上市日期"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT list_date FROM stock_info WHERE stock_code = ?",
                (stock_code,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
        return None
    
    def save_stock_info(
        self, 
        stock_code: str, 
        stock_name: str = None,
        list_date: str = None,
        industry: str = None
    ):
        """保存股票基本信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO stock_info 
                (stock_code, stock_name, list_date, industry, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (stock_code, stock_name, list_date, industry))
            conn.commit()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 股票数量
            cursor.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_daily")
            stock_count = cursor.fetchone()[0]
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM stock_daily")
            total_records = cursor.fetchone()[0]
            
            # 预测记录数
            cursor.execute("SELECT COUNT(*) FROM cycle_predictions")
            prediction_count = cursor.fetchone()[0]
            
            # 评估记录数
            cursor.execute("SELECT COUNT(*) FROM prediction_accuracy")
            accuracy_count = cursor.fetchone()[0]
            
            # 数据库大小
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'stock_count': stock_count,
                'total_records': total_records,
                'prediction_count': prediction_count,
                'accuracy_count': accuracy_count,
                'db_size_mb': round(db_size / 1024 / 1024, 2)
            }
    
    def delete_stock_data(self, stock_code: str):
        """删除某只股票的所有数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stock_daily WHERE stock_code = ?", (stock_code,))
            cursor.execute("DELETE FROM stock_info WHERE stock_code = ?", (stock_code,))
            cursor.execute("DELETE FROM update_log WHERE stock_code = ?", (stock_code,))
            cursor.execute("DELETE FROM cycle_predictions WHERE stock_code = ?", (stock_code,))
            cursor.execute("DELETE FROM prediction_accuracy WHERE stock_code = ?", (stock_code,))
            conn.commit()
            return cursor.rowcount
