"""
股票周期预测模块 - 基于HMM状态和历史数据的改进版
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from model_optimizer import ModelOptimizer


@dataclass
class CyclePrediction:
    """周期预测结果"""
    stock_code: str
    analysis_date: str
    current_stage: str
    
    # 周期时间预测
    cycle_start_date: str
    cycle_estimated_end: str
    cycle_progress: float
    days_in_cycle: int
    estimated_total_days: int
    
    # 基于历史统计的周期长度
    hist_avg_duration: int
    hist_min_duration: int
    hist_max_duration: int
    hist_sample_count: int  # 历史样本数
    
    # 价格预测
    current_price: float
    price_target_low: float
    price_target_high: float
    price_target_mean: float
    
    # 短期预测（5天、20天）
    pred_5d_stage: str
    pred_5d_price_low: float
    pred_5d_price_high: float
    pred_5d_confidence: float
    
    pred_20d_stage: str
    pred_20d_price_low: float
    pred_20d_price_high: float
    pred_20d_confidence: float
    
    # 预测依据
    reasoning: str
    key_indicators: Dict[str, float]


class CyclePredictor:
    """周期预测器 - 基于HMM状态和历史数据"""
    
    # 各阶段的价格变动预期
    STAGE_PRICE_CHANGE = {
        "吸筹期": {"5d": (-0.05, 0.08), "20d": (-0.10, 0.20)},
        "拉升期": {"5d": (0.02, 0.15), "20d": (0.10, 0.40)},
        "派发期": {"5d": (-0.08, 0.03), "20d": (-0.20, 0.05)},
        "观望期": {"5d": (-0.05, 0.05), "20d": (-0.10, 0.10)}
    }
    
    # 默认周期长度（当没有历史数据时使用）
    DEFAULT_DURATION = {
        "吸筹期": {"avg": 80, "min": 40, "max": 150},
        "拉升期": {"avg": 30, "min": 15, "max": 60},
        "派发期": {"avg": 50, "min": 25, "max": 90},
        "观望期": {"avg": 40, "min": 20, "max": 80}
    }
    
    def __init__(self):
        self.optimizer = ModelOptimizer()
    
    def _analyze_hmm_states(self, df: pd.DataFrame, hmm_result: Dict) -> List[Dict]:
        """
        基于HMM状态序列分析历史周期
        
        Returns:
            周期列表
        """
        cycles = []
        states = hmm_result.get('state_sequence', [])
        
        if not states:
            return cycles
        
        # 确保状态序列长度与数据长度一致
        if len(states) != len(df):
            # 如果长度不匹配，截断或填充
            if len(states) > len(df):
                states = states[:len(df)]
            else:
                # 填充最后一个状态
                states = states + [states[-1]] * (len(df) - len(states))
        
        # 获取状态到阶段的映射
        state_to_stage = self._infer_state_stages(df, states)
        
        # 合并连续的同状态区间
        current_state = states[0]
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] != current_state or i == len(states) - 1:
                # 结束当前周期
                end_idx = i if i == len(states) - 1 else i - 1
                duration = end_idx - start_idx + 1
                if duration >= 10:  # 至少10天
                    stage = state_to_stage.get(current_state, '观望期')
                    cycles.append({
                        'stage': stage,
                        'start_date': df['date'].iloc[start_idx],
                        'end_date': df['date'].iloc[end_idx],
                        'duration': duration,
                        'start_price': df['close'].iloc[start_idx],
                        'end_price': df['close'].iloc[end_idx],
                        'price_change': (df['close'].iloc[end_idx] / df['close'].iloc[start_idx] - 1)
                    })
                current_state = states[i]
                start_idx = i
        
        return cycles
    
    def _infer_state_stages(self, df: pd.DataFrame, states: List[int]) -> Dict[int, str]:
        """
        根据每个状态的价格特征推断对应的阶段
        """
        state_features = {}
        unique_states = set(states)
        
        for state_id in unique_states:
            mask = np.array(states) == state_id
            if mask.sum() > 0:
                state_df = df[mask]
                avg_price = state_df['close'].mean()
                avg_price_pos = (avg_price - df['close'].min()) / (df['close'].max() - df['close'].min() + 1e-6)
                # 计算该状态下的平均涨跌幅（日收益*20）
                daily_returns = state_df['close'].pct_change().dropna()
                avg_return = daily_returns.mean() * 20 if len(daily_returns) > 0 else 0
                # 计算价格变动（从状态开始到结束）
                price_change = (state_df['close'].iloc[-1] / state_df['close'].iloc[0] - 1)
                state_features[state_id] = {
                    'price_pos': avg_price_pos,
                    'avg_return': avg_return,
                    'price_change': price_change
                }
        
        # 根据特征分配阶段
        state_to_stage = {}
        
        for state_id, features in state_features.items():
            price_pos = features['price_pos']
            price_change = features['price_change']
            
            # 判断逻辑：
            # 1. 价格位置低 + 涨幅小 = 吸筹期
            # 2. 价格位置高 + 涨幅大 = 拉升期
            # 3. 价格位置高 + 跌幅 = 派发期
            # 4. 其他 = 观望期
            
            if price_pos < 0.4 and price_change < 0.15:
                stage = '吸筹期'
            elif price_pos > 0.6 and price_change > 0.10:
                stage = '拉升期'
            elif price_pos > 0.5 and price_change < -0.05:
                stage = '派发期'
            elif abs(price_change) < 0.05:
                stage = '观望期'
            else:
                # 根据价格位置默认分配
                if price_pos < 0.35:
                    stage = '吸筹期'
                elif price_pos > 0.75:
                    stage = '派发期'
                elif price_change > 0:
                    stage = '拉升期'
                else:
                    stage = '观望期'
            
            state_to_stage[state_id] = stage
        
        return state_to_stage
    
    def _find_current_cycle_start(self, df: pd.DataFrame, hmm_result: Dict) -> Tuple[pd.Timestamp, int]:
        """
        查找当前周期的开始时间和已持续天数
        
        Returns:
            (开始日期, 已持续天数)
        """
        states = hmm_result.get('state_sequence', [])
        current_state_id = hmm_result.get('state_id', 0)
        
        if not states or len(states) != len(df):
            # 没有HMM状态序列，使用价格特征估算
            return self._estimate_cycle_start_by_price(df, hmm_result)
        
        # 从后往前找当前状态的连续区间
        end_idx = len(states) - 1
        start_idx = end_idx
        
        for i in range(end_idx, -1, -1):
            if states[i] == current_state_id:
                start_idx = i
            else:
                break
        
        days_in_cycle = end_idx - start_idx + 1
        cycle_start = df['date'].iloc[start_idx]
        
        return cycle_start, days_in_cycle
    
    def _estimate_cycle_start_by_price(self, df: pd.DataFrame, hmm_result: Dict) -> Tuple[pd.Timestamp, int]:
        """
        基于价格特征估算周期开始时间
        """
        current_stage = hmm_result.get('current_stage', '观望期')
        recent_df = df.tail(120)  # 看最近120天
        
        if current_stage == '吸筹期':
            # 找近期最低点
            min_idx = recent_df['low'].idxmin()
            days_in_cycle = len(df) - df.index.get_loc(min_idx)
            return df.loc[min_idx, 'date'], days_in_cycle
        
        elif current_stage == '拉升期':
            # 找突破点或最低点
            if 'ma20' in recent_df.columns:
                breakout = recent_df[
                    (recent_df['close'] > recent_df['ma20'] * 1.03) &
                    (recent_df['close'].diff(5) > 0)
                ]
                if not breakout.empty:
                    start_idx = df.index.get_loc(breakout.index[0])
                    days_in_cycle = len(df) - start_idx
                    return breakout.iloc[0]['date'], days_in_cycle
            # 找不到突破点，找最低点
            min_idx = recent_df['low'].idxmin()
            days_in_cycle = len(df) - df.index.get_loc(min_idx)
            return df.loc[min_idx, 'date'], days_in_cycle
        
        elif current_stage == '派发期':
            # 找近期最高点
            max_idx = recent_df['high'].idxmax()
            days_in_cycle = len(df) - df.index.get_loc(max_idx)
            return df.loc[max_idx, 'date'], days_in_cycle
        
        else:  # 观望期
            # 找趋势转折点或固定30天
            return df['date'].iloc[-30], 30
    
    def _calculate_stage_stats(self, cycles: List[Dict], stage: str) -> Dict:
        """
        计算某阶段的历史统计
        """
        stage_cycles = [c for c in cycles if c['stage'] == stage]
        
        if len(stage_cycles) == 0:
            defaults = self.DEFAULT_DURATION.get(stage, {"avg": 60, "min": 30, "max": 90})
            return {
                "avg": defaults["avg"],
                "min": defaults["min"],
                "max": defaults["max"],
                "count": 0
            }
        
        durations = [c['duration'] for c in stage_cycles]
        return {
            "avg": int(np.mean(durations)),
            "min": int(np.min(durations)),
            "max": int(np.max(durations)),
            "count": len(durations)
        }
    
    def predict(
        self,
        stock_code: str,
        df: pd.DataFrame,
        hmm_result: Dict,
        dna=None
    ) -> CyclePrediction:
        """
        基于HMM结果和历史数据进行周期预测
        """
        current_stage = hmm_result['current_stage']
        current_price = df['close'].iloc[-1]
        analysis_date = df['date'].iloc[-1]
        
        # 1. 查找当前周期开始时间
        cycle_start, days_in_cycle = self._find_current_cycle_start(df, hmm_result)
        
        # 2. 分析历史周期（基于HMM状态）
        historical_cycles = self._analyze_hmm_states(df, hmm_result)
        
        # 3. 计算历史统计
        stats = self._calculate_stage_stats(historical_cycles, current_stage)
        
        # 4. 计算周期进度和预估总长度
        if stats['count'] > 0:
            # 有历史数据
            hist_avg = stats['avg']
            # 如果当前已超过历史平均，动态调整预估
            if days_in_cycle > hist_avg:
                # 已超出历史平均，按当前进度估算
                estimated_total = int(days_in_cycle / 0.7)  # 假设已完成70%
                # 但不超出历史最长的1.5倍
                estimated_total = min(estimated_total, int(stats['max'] * 1.5))
            else:
                estimated_total = hist_avg
        else:
            # 无历史数据，使用默认值
            estimated_total = stats['avg']
        
        cycle_progress = min(days_in_cycle / estimated_total, 0.95)
        
        # 5. 预估周期结束时间
        remaining_days = max(0, int(estimated_total - days_in_cycle))
        cycle_end = analysis_date + timedelta(days=remaining_days)
        
        # 6. 价格目标预测
        price_target = self._predict_price_target(
            df, current_stage, current_price, cycle_progress, historical_cycles
        )
        
        # 7. 短期预测
        pred_5d = self._predict_short_term(df, current_stage, current_price, 5, cycle_progress)
        pred_20d = self._predict_short_term(df, current_stage, current_price, 20, cycle_progress)
        
        # 8. 生成预测依据
        reasoning = self._generate_reasoning(
            current_stage, cycle_progress, days_in_cycle, 
            estimated_total, stats, price_target
        )
        
        # 9. 关键指标
        key_indicators = {
            'volatility_20d': df['volatility'].iloc[-20:].mean() if 'volatility' in df.columns else 0,
            'volume_ratio': df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean() if 'volume' in df.columns else 1,
            'price_position': df['price_position'].iloc[-1] if 'price_position' in df.columns else 0.5,
            'price_vs_hist_high': current_price / df['high'].max(),
            'cycle_progress': cycle_progress,
            'hist_cycles_count': stats['count'],
            'hist_avg_duration': stats['avg']
        }
        
        return CyclePrediction(
            stock_code=stock_code,
            analysis_date=analysis_date.strftime('%Y-%m-%d'),
            current_stage=current_stage,
            cycle_start_date=cycle_start.strftime('%Y-%m-%d'),
            cycle_estimated_end=cycle_end.strftime('%Y-%m-%d'),
            cycle_progress=cycle_progress,
            days_in_cycle=days_in_cycle,
            estimated_total_days=estimated_total,
            hist_avg_duration=stats['avg'],
            hist_min_duration=stats['min'],
            hist_max_duration=stats['max'],
            hist_sample_count=stats['count'],
            current_price=round(current_price, 2),
            price_target_low=round(price_target['low'], 2),
            price_target_high=round(price_target['high'], 2),
            price_target_mean=round(price_target['mean'], 2),
            pred_5d_stage=pred_5d['stage'],
            pred_5d_price_low=round(pred_5d['price_low'], 2),
            pred_5d_price_high=round(pred_5d['price_high'], 2),
            pred_5d_confidence=pred_5d['confidence'],
            pred_20d_stage=pred_20d['stage'],
            pred_20d_price_low=round(pred_20d['price_low'], 2),
            pred_20d_price_high=round(pred_20d['price_high'], 2),
            pred_20d_confidence=pred_20d['confidence'],
            reasoning=reasoning,
            key_indicators=key_indicators
        )
    
    def _predict_price_target(
        self, 
        df: pd.DataFrame, 
        stage: str, 
        current_price: float,
        cycle_progress: float,
        historical_cycles: List[Dict] = None
    ) -> Dict:
        """预测周期结束时的价格目标"""
        
        # 如果有历史同阶段数据，参考历史价格变动
        if historical_cycles:
            same_stage_cycles = [c for c in historical_cycles if c['stage'] == stage]
            if same_stage_cycles:
                avg_price_change = np.mean([c['price_change'] for c in same_stage_cycles])
                remaining_change = avg_price_change * (1 - cycle_progress)
                target_mean = current_price * (1 + remaining_change)
                return {
                    'low': target_mean * 0.9,
                    'high': target_mean * 1.1,
                    'mean': target_mean
                }
        
        # 默认预测逻辑
        if stage == '吸筹期':
            potential = 0.15 + 0.20 * (1 - cycle_progress)
            low = current_price * (1 + potential * 0.3)
            high = current_price * (1 + potential)
            mean = current_price * (1 + potential * 0.7)
        elif stage == '拉升期':
            remaining_up = 0.08 * (1 - cycle_progress)
            low = current_price * 0.90
            high = current_price * (1 + remaining_up)
            mean = current_price * 0.95
        elif stage == '派发期':
            remaining_down = 0.12 * (1 - cycle_progress)
            low = current_price * (1 - remaining_down)
            high = current_price * 1.02
            mean = current_price * 0.95
        else:
            low = current_price * 0.95
            high = current_price * 1.05
            mean = current_price
        
        return {'low': low, 'high': high, 'mean': mean}
    
    def _predict_short_term(
        self, 
        df: pd.DataFrame, 
        current_stage: str, 
        current_price: float,
        days: int,
        cycle_progress: float = 0.5
    ) -> Dict:
        """短期价格预测"""
        
        key = f"{days}d"
        change_range = self.STAGE_PRICE_CHANGE.get(current_stage, {}).get(key, (-0.05, 0.05))
        
        recent_return = (df['close'].iloc[-1] - df['close'].iloc[-min(days, len(df)):].iloc[0]) / df['close'].iloc[-min(days, len(df)):].iloc[0]
        
        adjusted_low = min(change_range[0], recent_return * 0.8)
        adjusted_high = max(change_range[1], recent_return * 1.2)
        
        if current_stage == '吸筹期':
            next_stage = '拉升期' if cycle_progress > 0.7 else '吸筹期'
        elif current_stage == '拉升期':
            next_stage = '派发期' if cycle_progress > 0.6 else '拉升期'
        elif current_stage == '派发期':
            next_stage = '观望期' if cycle_progress > 0.7 else '派发期'
        else:
            next_stage = '吸筹期'
        
        confidence = 0.6 + 0.3 * (1 - abs(cycle_progress - 0.5))
        
        return {
            'stage': next_stage,
            'price_low': current_price * (1 + adjusted_low),
            'price_high': current_price * (1 + adjusted_high),
            'confidence': round(confidence, 2)
        }
    
    def _generate_reasoning(
        self, 
        stage: str, 
        progress: float, 
        days_in: int,
        estimated_total: int,
        stats: Dict,
        price_target: Dict
    ) -> str:
        """生成预测依据说明"""
        
        progress_desc = "初期" if progress < 0.3 else "中期" if progress < 0.7 else "后期"
        
        reasoning = f"当前处于{stage}{progress_desc}，已持续{days_in}天。"
        
        # 添加历史统计信息
        if stats['count'] > 0:
            reasoning += f"根据该股票历史数据，{stage}平均持续{stats['avg']}天（范围{stats['min']}-{stats['max']}天，基于{stats['count']}个样本）。"
            
            if days_in > stats['max']:
                reasoning += f"⚠️ 注意：当前已持续天数超过历史最长，可能处于特殊行情或阶段判断有误。"
            elif days_in > stats['avg'] * 1.2:
                reasoning += f"当前已超过历史平均，可能接近尾声。"
            elif days_in < stats['min']:
                reasoning += f"当前尚处于历史最短周期内，可能还需时间。"
        else:
            reasoning += f"该股票缺乏{stage}历史数据，使用默认周期估算（{estimated_total}天）。"
        
        # 阶段-specific说明
        if stage == '吸筹期':
            if progress < 0.3:
                reasoning += "吸筹刚开始，主力建仓尚不充分。"
            elif progress < 0.7:
                reasoning += "吸筹进行中，关注成交量变化和筹码集中迹象。"
            else:
                reasoning += "吸筹可能接近尾声，密切关注放量突破信号。"
        elif stage == '拉升期':
            if progress < 0.5:
                reasoning += "拉升初期，上涨动能相对充足。"
            else:
                reasoning += "拉升已进行一段时间，注意高位风险。"
        elif stage == '派发期':
            if progress < 0.5:
                reasoning += "派发初期，主力出货可能尚不充分。"
            else:
                reasoning += "派发后期，建议观望等待下一周期。"
        
        reasoning += f"预计周期结束时价格区间：{price_target['low']:.2f}-{price_target['high']:.2f}元。"
        
        return reasoning
