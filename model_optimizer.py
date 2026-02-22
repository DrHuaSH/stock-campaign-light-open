"""
模型参数优化模块
基于历史预测准确性反馈，自动优化周期预测模型参数
"""
import sqlite3
import json
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from config import DATA_DIR


@dataclass
class StageDurationParams:
    """阶段持续时间参数"""
    min_days: int
    max_days: int
    typical_days: int
    
    def adjust(self, factor: float):
        """根据反馈调整参数"""
        self.typical_days = int(self.typical_days * factor)
        self.typical_days = max(self.min_days, min(self.max_days, self.typical_days))


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self):
        self.db_path = DATA_DIR / "stock_data.db"
        self.config_path = DATA_DIR / "model_params.json"
        self._load_params()
    
    def _load_params(self):
        """加载模型参数"""
        default_params = {
            "吸筹期": {"min": 60, "max": 180, "typical": 120, "price_potential": 0.25},
            "拉升期": {"min": 20, "max": 60, "typical": 40, "remaining_up": 0.10},
            "派发期": {"min": 40, "max": 120, "typical": 80, "remaining_down": 0.15},
            "观望期": {"min": 30, "max": 90, "typical": 60, "volatility": 0.05}
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.params = json.load(f)
        else:
            self.params = default_params
            self._save_params()
    
    def _save_params(self):
        """保存模型参数"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, ensure_ascii=False, indent=2)
    
    def get_stage_duration(self, stage: str) -> Dict:
        """获取阶段持续时间参数"""
        return self.params.get(stage, {"min": 30, "max": 90, "typical": 60})
    
    def analyze_accuracy_trends(self, stock_code: str = None) -> Dict:
        """
        分析预测准确性趋势
        
        Returns:
            各阶段、各预测类型的准确性统计
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 构建查询
            if stock_code:
                query = """
                    SELECT 
                        p.current_stage,
                        a.prediction_type,
                        AVG(a.accuracy_score) as avg_accuracy,
                        AVG(a.price_error_rate) as avg_price_error,
                        AVG(a.stage_correct) as stage_correct_rate,
                        COUNT(*) as count
                    FROM prediction_accuracy a
                    JOIN cycle_predictions p ON a.prediction_id = p.id
                    WHERE a.stock_code = ?
                    GROUP BY p.current_stage, a.prediction_type
                """
                cursor.execute(query, (stock_code,))
            else:
                query = """
                    SELECT 
                        p.current_stage,
                        a.prediction_type,
                        AVG(a.accuracy_score) as avg_accuracy,
                        AVG(a.price_error_rate) as avg_price_error,
                        AVG(a.stage_correct) as stage_correct_rate,
                        COUNT(*) as count
                    FROM prediction_accuracy a
                    JOIN cycle_predictions p ON a.prediction_id = p.id
                    GROUP BY p.current_stage, a.prediction_type
                """
                cursor.execute(query)
            
            results = cursor.fetchall()
            
            # 整理结果
            trends = {}
            for row in results:
                stage, pred_type, accuracy, price_error, stage_correct, count = row
                if stage not in trends:
                    trends[stage] = {}
                trends[stage][pred_type] = {
                    'avg_accuracy': accuracy,
                    'avg_price_error': price_error,
                    'stage_correct_rate': stage_correct,
                    'sample_count': count
                }
            
            return trends
    
    def suggest_parameter_adjustments(self) -> List[Dict]:
        """
        根据准确性分析，建议参数调整
        
        Returns:
            参数调整建议列表
        """
        trends = self.analyze_accuracy_trends()
        suggestions = []
        
        for stage, type_stats in trends.items():
            for pred_type, stats in type_stats.items():
                accuracy = stats['avg_accuracy']
                price_error = stats['avg_price_error']
                count = stats['sample_count']
                
                # 样本量不足时不建议调整
                if count < 5:
                    continue
                
                # 准确率低于 60% 需要调整
                if accuracy < 0.6:
                    if pred_type == 'cycle_end':
                        # 周期结束预测不准，调整持续时间参数
                        current_typical = self.params[stage]['typical']
                        
                        # 如果价格误差为正（预测价格偏低），说明周期可能比预期长
                        if price_error > 0.1:
                            suggested = int(current_typical * 1.1)
                            suggestions.append({
                                'stage': stage,
                                'parameter': 'typical_days',
                                'current_value': current_typical,
                                'suggested_value': suggested,
                                'reason': f'周期结束预测准确率仅{accuracy:.1%}，价格误差{price_error:.1%}，建议延长周期预期',
                                'confidence': min(count / 20, 1.0)  # 样本越多置信度越高
                            })
                        # 如果价格误差为负（预测价格偏高），说明周期可能比预期短
                        elif price_error < -0.1:
                            suggested = int(current_typical * 0.9)
                            suggestions.append({
                                'stage': stage,
                                'parameter': 'typical_days',
                                'current_value': current_typical,
                                'suggested_value': suggested,
                                'reason': f'周期结束预测准确率仅{accuracy:.1%}，价格误差{price_error:.1%}，建议缩短周期预期',
                                'confidence': min(count / 20, 1.0)
                            })
                    
                    elif pred_type in ['5d', '20d']:
                        # 短期预测不准，可能需要调整价格波动参数
                        suggestions.append({
                            'stage': stage,
                            'parameter': f'{pred_type}_price_range',
                            'current_value': 'default',
                            'suggested_value': 'widen',
                            'reason': f'{pred_type}预测准确率仅{accuracy:.1%}，建议放宽价格波动区间',
                            'confidence': min(count / 20, 1.0)
                        })
        
        # 按置信度排序
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions
    
    def apply_adjustments(self, suggestions: List[Dict], dry_run: bool = True) -> Dict:
        """
        应用参数调整建议
        
        Args:
            suggestions: 调整建议列表
            dry_run: 如果为 True，只返回预览不实际应用
            
        Returns:
            应用结果
        """
        applied = []
        skipped = []
        
        for suggestion in suggestions:
            if suggestion['confidence'] < 0.5:
                skipped.append({
                    **suggestion,
                    'reason': '置信度不足'
                })
                continue
            
            stage = suggestion['stage']
            param = suggestion['parameter']
            new_value = suggestion['suggested_value']
            
            if dry_run:
                applied.append({
                    **suggestion,
                    'status': 'preview'
                })
            else:
                # 实际应用调整
                if param == 'typical_days':
                    self.params[stage]['typical'] = new_value
                    applied.append({
                        **suggestion,
                        'status': 'applied'
                    })
        
        if not dry_run:
            self._save_params()
        
        return {
            'applied': applied,
            'skipped': skipped,
            'dry_run': dry_run
        }
    
    def get_optimization_report(self) -> Dict:
        """
        生成模型优化报告
        
        Returns:
            包含准确性统计、参数调整建议等的完整报告
        """
        trends = self.analyze_accuracy_trends()
        suggestions = self.suggest_parameter_adjustments()
        
        # 计算整体统计
        total_samples = 0
        total_accuracy = 0
        
        for stage_stats in trends.values():
            for type_stats in stage_stats.values():
                count = type_stats['sample_count']
                total_samples += count
                total_accuracy += type_stats['avg_accuracy'] * count
        
        overall_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_evaluated_predictions': total_samples,
            'accuracy_by_stage': trends,
            'parameter_suggestions': suggestions,
            'current_params': self.params
        }
    
    def reset_to_defaults(self):
        """重置参数为默认值"""
        default_params = {
            "吸筹期": {"min": 60, "max": 180, "typical": 120, "price_potential": 0.25},
            "拉升期": {"min": 20, "max": 60, "typical": 40, "remaining_up": 0.10},
            "派发期": {"min": 40, "max": 120, "typical": 80, "remaining_down": 0.15},
            "观望期": {"min": 30, "max": 90, "typical": 60, "volatility": 0.05}
        }
        self.params = default_params
        self._save_params()
        return default_params
