"""
HMM股票分析核心模块 - 改进版
基于特征明确标注阶段，避免无监督学习的状态混乱问题
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from hmmlearn import hmm

from config import HMM_CONFIG, STAGE_NAMES, STAGE_NAMES_EN


@dataclass
class StockDNA:
    """股票DNA - 包含HMM模型参数和特征模板"""
    stock_code: str
    stock_name: str
    created_at: str
    updated_at: str
    
    # HMM模型参数
    n_states: int
    transition_matrix: List[List[float]]
    means: List[List[float]]
    covars: List[List[float]]
    
    # 特征模板（各阶段的典型特征）
    stage_templates: Dict[str, Dict[str, float]]
    
    # 历史统计
    total_days: int
    stage_distribution: Dict[str, float]
    
    # 价格参考（用于判断相对位置）
    price_stats: Dict[str, float]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StockDNA':
        return cls(**data)


class HMMAnalyzer:
    """HMM股票分析器 - 改进版"""
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model = None
        self.is_trained = False
        self.feature_names = [
            'returns', 'volume_change', 'volatility', 
            'price_position', 'trend_strength', 'volume_ratio'
        ]
        
        # 阶段特征定义（用于标注HMM状态）
        self.stage_characteristics = {
            '吸筹期': {
                'returns': (-0.02, 0.02),      # 涨跌幅小
                'volatility': (0.01, 0.03),     # 波动率低
                'price_position': (0.1, 0.4),   # 价格在相对低位
                'trend_strength': (-0.02, 0.02), # 无明显趋势
                'volume_ratio': (0.8, 1.5),     # 成交量温和
            },
            '拉升期': {
                'returns': (0.01, 0.05),        # 涨跌幅为正
                'volatility': (0.02, 0.05),     # 波动率中等偏高
                'price_position': (0.5, 0.9),   # 价格在相对高位
                'trend_strength': (0.03, 0.10), # 强上升趋势
                'volume_ratio': (1.2, 3.0),     # 成交量放大
            },
            '派发期': {
                'returns': (-0.03, 0.02),       # 涨跌幅可能为正但减弱
                'volatility': (0.02, 0.05),     # 波动率高
                'price_position': (0.6, 0.95),  # 价格在高位
                'trend_strength': (-0.05, 0.02), # 趋势减弱或转弱
                'volume_ratio': (1.0, 2.5),     # 成交量仍大但可能萎缩
            },
            '观望期': {
                'returns': (-0.02, 0.02),       # 涨跌幅小
                'volatility': (0.01, 0.025),    # 波动率低
                'price_position': (0.3, 0.7),   # 价格在中位
                'trend_strength': (-0.02, 0.02), # 无明显趋势
                'volume_ratio': (0.5, 1.0),     # 成交量萎缩
            }
        }
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从股票数据提取特征 - 改进版
        
        新增特征:
        - trend_strength: 趋势强度（20日斜率标准化）
        - price_relative_to_high: 相对历史高点的位置
        """
        features = pd.DataFrame()
        
        # 基础特征
        features['returns'] = df['returns'].fillna(0)
        features['volume_change'] = df['volume_change'].fillna(0)
        features['volatility'] = df['volatility'].fillna(df['volatility'].median())
        features['price_position'] = df['price_position'].fillna(0.5)
        
        # 趋势强度 - 20日价格线性回归斜率
        if len(df) >= 20:
            ma20 = df['close'].rolling(window=20, min_periods=10).mean()
            # 计算斜率并标准化
            trend = ma20.diff(5) / ma20.shift(5)
            features['trend_strength'] = trend.fillna(0)
        else:
            features['trend_strength'] = 0
        
        # 成交量比
        features['volume_ratio'] = (df['volume'] / df['volume_ma5']).fillna(1)
        
        # 相对历史高点的位置（用于判断是否在高位）
        if len(df) >= 60:
            high_60 = df['high'].rolling(60, min_periods=20).max()
            features['price_relative_to_high'] = (df['close'] / high_60).fillna(0.5)
        else:
            features['price_relative_to_high'] = 0.5
        
        # 处理异常值
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        return features
    
    def _classify_stage_by_features(self, features: pd.Series, df: pd.DataFrame = None) -> str:
        """
        根据特征直接判断阶段 - 规则引擎
        用于标注HMM状态和修正错误判断
        
        关键改进：更重视价格相对于历史高点的位置，而不是仅看20日区间
        """
        returns = features['returns']
        volatility = features['volatility']
        price_pos_20d = features['price_position']  # 20日相对位置
        trend = features['trend_strength']
        vol_ratio = features['volume_ratio']
        price_to_high_60d = features.get('price_relative_to_high', 0.5)  # 相对60日高点
        
        # 数据充足性检查
        has_enough_data = df is not None and len(df) >= 40
        
        # 如果有足够数据，计算历史统计
        if has_enough_data:
            hist_high = df['high'].max()
            hist_low = df['low'].min()
            current_price = df['close'].iloc[-1]
            price_vs_hist_high = current_price / hist_high if hist_high > 0 else 0.5
            
            # 计算近期趋势（10日 vs 30日）
            if len(df) >= 30:
                recent_ma10 = df['close'].tail(10).mean()
                recent_ma30 = df['close'].tail(30).mean()
                medium_trend = (recent_ma10 - recent_ma30) / recent_ma30 if recent_ma30 > 0 else 0
            else:
                medium_trend = trend
        else:
            # 数据不足时，使用20日位置和60日高点作为参考
            price_vs_hist_high = price_to_high_60d
            medium_trend = trend
        
        # ========== 数据不足时的简化判断 ==========
        if not has_enough_data:
            # 主要依赖20日相对位置和趋势
            if price_pos_20d > 0.6 and trend > 0.01:
                return '拉升期'
            elif price_pos_20d < 0.3 and trend < -0.01:
                return '派发期'
            elif price_pos_20d < 0.4 and abs(trend) < 0.02:
                return '吸筹期'
            else:
                return '观望期'
        
        # ========== 数据充足时的核心判断逻辑 ==========
        
        # ===== 情况1：价格接近历史高点 (>75%) =====
        if price_vs_hist_high > 0.75:
            # 在高位时，判断是拉升期还是派发期取决于趋势
            if trend > 0.015 or medium_trend > 0.01:
                # 趋势仍向上，还在拉升
                return '拉升期'
            elif trend < -0.01 or medium_trend < -0.005 or returns < -0.015:
                # 趋势转弱或下跌，进入派发
                return '派发期'
            elif volatility > 0.025:
                # 高位震荡，波动大，可能是派发
                return '派发期'
            else:
                # 高位横盘，可能是拉升末期或派发初期
                return '拉升期' if vol_ratio > 1.2 else '派发期'
        
        # ===== 情况2：价格在历史低位 (<40%) =====
        elif price_vs_hist_high < 0.40:
            # 在低位时，判断是吸筹期还是观望期
            if volatility < 0.025 and abs(trend) < 0.025:
                # 低波动，无明显趋势，可能是吸筹或观望
                # 吸筹通常伴随温和放量，观望则是缩量
                if vol_ratio > 0.75 or price_vs_hist_high < 0.30:
                    return '吸筹期'  # 有成交量或价格很低，可能是吸筹
                else:
                    return '观望期'  # 无量，观望
            elif trend > 0.015:
                # 低位开始上涨，可能是吸筹后期或拉升初期
                return '吸筹期'
            else:
                return '观望期'
        
        # ===== 情况3：价格在中位 (40%-75%) =====
        else:
            # 中位时主要看趋势
            if trend > 0.025 or medium_trend > 0.02:
                return '拉升期'
            elif trend < -0.02 or medium_trend < -0.015:
                return '派发期'
            elif abs(trend) < 0.015 and volatility < 0.02:
                return '观望期'
            else:
                # 根据20日位置辅助判断
                if price_pos_20d > 0.6:
                    return '拉升期'
                elif price_pos_20d < 0.3:
                    return '派发期'
                else:
                    return '观望期'
    
    def _map_hmm_states_to_stages(self, features_df: pd.DataFrame, states: np.ndarray, df: pd.DataFrame = None) -> Dict[int, str]:
        """
        将HMM状态映射到阶段名称
        通过分析每个状态的特征均值来确定对应阶段
        """
        state_to_stage = {}
        
        for state_id in range(self.n_states):
            mask = states == state_id
            if mask.sum() > 0:
                state_features = features_df[mask].mean()
                stage = self._classify_stage_by_features(state_features, df)
                state_to_stage[state_id] = stage
        
        return state_to_stage
    
    def train(self, df: pd.DataFrame) -> StockDNA:
        """训练HMM模型并生成股票DNA"""
        features_df = self.extract_features(df)
        features = features_df[self.feature_names].values
        
        # 创建并训练HMM模型
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=HMM_CONFIG["n_iter"],
            random_state=42
        )
        
        self.model.fit(features)
        self.is_trained = True
        
        # 预测状态序列
        states = self.model.predict(features)
        
        # 将HMM状态映射到阶段
        self.state_to_stage = self._map_hmm_states_to_stages(features_df, states, df)
        
        # 构建股票DNA
        dna = self._build_dna(df, features_df, features, states)
        
        return dna
    
    def _build_dna(
        self, 
        df: pd.DataFrame, 
        features_df: pd.DataFrame,
        features: np.ndarray, 
        states: np.ndarray
    ) -> StockDNA:
        """构建股票DNA"""
        from datetime import datetime
        
        stock_code = df.get('ts_code', ['unknown'])[0] if 'ts_code' in df.columns else 'unknown'
        
        # 计算各阶段的特征模板
        stage_templates = {}
        for state_id in range(self.n_states):
            mask = states == state_id
            if mask.sum() > 0:
                state_features = features[mask]
                template = {
                    name: float(np.mean(state_features[:, i]))
                    for i, name in enumerate(self.feature_names)
                }
                stage_name = self.state_to_stage.get(state_id, f"state_{state_id}")
                stage_templates[stage_name] = template
        
        # 计算阶段分布（按阶段名称统计）
        stage_counts = {'吸筹期': 0, '拉升期': 0, '派发期': 0, '观望期': 0}
        for state_id, count in zip(*np.unique(states, return_counts=True)):
            stage_name = self.state_to_stage.get(state_id, 'unknown')
            if stage_name in stage_counts:
                stage_counts[stage_name] += count
        
        total = len(states)
        stage_distribution = {k: v/total for k, v in stage_counts.items()}
        
        # 价格统计
        price_stats = {
            'min': float(df['close'].min()),
            'max': float(df['close'].max()),
            'mean': float(df['close'].mean()),
            'current': float(df['close'].iloc[-1]),
            'percentile': float((df['close'].iloc[-1] - df['close'].min()) / (df['close'].max() - df['close'].min() + 1e-6))
        }
        
        return StockDNA(
            stock_code=stock_code,
            stock_name="",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            n_states=self.n_states,
            transition_matrix=self.model.transmat_.tolist(),
            means=self.model.means_.tolist(),
            covars=self.model.covars_.tolist(),
            stage_templates=stage_templates,
            total_days=len(df),
            stage_distribution=stage_distribution,
            price_stats=price_stats
        )
    
    def analyze_current_stage(
        self, 
        df: pd.DataFrame,
        dna: Optional[StockDNA] = None
    ) -> Dict:
        """分析当前阶段 - 改进版"""
        if not self.is_trained and dna is None:
            raise ValueError("模型未训练，请先train()或提供DNA")
        
        # 如果提供了DNA，加载模型参数
        if dna and not self.is_trained:
            self._load_from_dna(dna)
        
        # 提取特征
        features_df = self.extract_features(df)
        features = features_df[self.feature_names].values
        
        # 预测状态
        states = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # 当前状态（最后一天）
        current_state = int(states[-1])
        current_probs = probabilities[-1]
        
        # 获取当前阶段名称
        current_stage = self.state_to_stage.get(current_state, 'unknown')
        
        # 基于规则引擎进行二次判断和修正
        latest_features = features_df.iloc[-1]
        rule_based_stage = self._classify_stage_by_features(latest_features, df)
        
        # 决策逻辑：结合HMM和规则引擎
        # 1. 数据不足40天时，优先使用规则引擎
        # 2. HMM置信度低时，使用规则引擎
        # 3. 两者差异大时，根据置信度决定
        hmm_confidence = current_probs[current_state]
        
        if len(df) < 40:
            # 数据不足，优先使用规则引擎
            current_stage = rule_based_stage
        elif current_stage != rule_based_stage:
            if hmm_confidence < 0.5:
                # HMM不确定，使用规则引擎
                current_stage = rule_based_stage
            elif hmm_confidence > 0.8 and current_stage != 'unknown':
                # HMM很确定，保持HMM判断
                pass
            else:
                # 中等置信度，如果规则引擎判断为派发期（重要信号），优先采用
                if rule_based_stage == '派发期':
                    current_stage = rule_based_stage
                # 否则保持HMM判断
        
        # 最近5天的状态稳定性
        recent_states = states[-5:] if len(states) >= 5 else states
        stability = np.mean(recent_states == current_state)
        
        # 置信度计算
        confidence = current_probs[current_state] * 0.5 + stability * 0.3 + 0.2
        confidence = min(confidence, 0.99)
        
        # 分析特征
        feature_analysis = {
            name: float(latest_features[name])
            for name in self.feature_names
        }
        
        # 与DNA模板比较（如果有）
        template_similarity = {}
        if dna and dna.stage_templates:
            for stage_name, template in dna.stage_templates.items():
                template_values = [template.get(name, 0) for name in self.feature_names]
                current_values = [latest_features[name] for name in self.feature_names]
                similarity = 1 - np.mean(np.abs(np.array(current_values) - np.array(template_values)))
                template_similarity[stage_name] = float(similarity)
        
        # 计算各阶段的概率（合并HMM概率和规则判断）
        all_stage_probs = {}
        for i in range(self.n_states):
            stage_name = self.state_to_stage.get(i, f'state_{i}')
            prob = current_probs[i]
            all_stage_probs[stage_name] = all_stage_probs.get(stage_name, 0) + prob
        
        # 确保所有阶段都有概率值
        for stage in ['吸筹期', '拉升期', '派发期', '观望期']:
            if stage not in all_stage_probs:
                all_stage_probs[stage] = 0.05
        
        # 归一化
        total_prob = sum(all_stage_probs.values())
        all_stage_probs = {k: v/total_prob for k, v in all_stage_probs.items()}
        
        return {
            "current_stage": current_stage,
            "current_stage_en": STAGE_NAMES_EN.get(list(STAGE_NAMES.keys())[list(STAGE_NAMES.values()).index(current_stage)], 'unknown') if current_stage in STAGE_NAMES.values() else 'unknown',
            "state_id": current_state,
            "confidence": float(confidence),
            "state_probability": float(current_probs[current_state]),
            "stability": float(stability),
            "feature_analysis": feature_analysis,
            "template_similarity": template_similarity,
            "all_stage_probs": all_stage_probs,
            "rule_based_stage": rule_based_stage,  # 规则引擎判断结果
            "price_stats": dna.price_stats if dna else {}
        }
    
    def _load_from_dna(self, dna: StockDNA):
        """从DNA加载模型参数"""
        self.model = hmm.GaussianHMM(
            n_components=dna.n_states,
            covariance_type="diag",
            n_iter=1,
            random_state=42
        )
        self.model.transmat_ = np.array(dna.transition_matrix)
        self.model.means_ = np.array(dna.means)
        
        # 处理 covars 维度问题
        # 对于 diag 协方差类型，covars 应该是 (n_components, n_dim)
        covars = np.array(dna.covars)
        n_dim = self.model.means_.shape[1]
        
        # 如果 covars 维度不对，进行调整
        if covars.ndim == 1:
            # 如果是一维数组，可能是 (n_components,)，需要扩展为 (n_components, n_dim)
            covars = np.tile(covars.reshape(-1, 1), (1, n_dim))
        elif covars.shape != (dna.n_states, n_dim):
            # 如果形状不匹配，重新初始化
            covars = np.ones((dna.n_states, n_dim))
        
        self.model.covars_ = covars
        self.n_states = dna.n_states
        
        # 重建状态到阶段的映射
        # 根据模板特征重新分类
        self.state_to_stage = {}
        for i in range(dna.n_states):
            # 找到最接近的模板
            best_stage = '观望期'
            best_score = -1
            for stage_name, template in dna.stage_templates.items():
                if stage_name in ['吸筹期', '拉升期', '派发期', '观望期']:
                    template_values = np.array([template.get(name, 0) for name in self.feature_names])
                    mean_values = dna.means[i] if i < len(dna.means) else template_values
                    similarity = 1 - np.mean(np.abs(mean_values - template_values))
                    if similarity > best_score:
                        best_score = similarity
                        best_stage = stage_name
            self.state_to_stage[i] = best_stage
        
        self.is_trained = True
    
    def batch_analyze(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        dna_dict: Dict[str, StockDNA] = None
    ) -> Dict[str, Dict]:
        """批量分析多只股票"""
        results = {}
        dna_dict = dna_dict or {}
        
        for stock_code, df in stock_data_dict.items():
            try:
                dna = dna_dict.get(stock_code)
                if dna:
                    self._load_from_dna(dna)
                else:
                    self.train(df)
                
                result = self.analyze_current_stage(df)
                result['stock_code'] = stock_code
                result['has_dna'] = dna is not None
                results[stock_code] = result
                
            except Exception as e:
                results[stock_code] = {
                    'stock_code': stock_code,
                    'error': str(e)
                }
        
        return results
