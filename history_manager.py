"""
分析历史记录管理模块
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from config import HISTORY_DIR


@dataclass
class AnalysisRecord:
    """分析记录"""
    record_id: str  # 唯一ID
    analysis_time: str
    stock_code: str
    stock_name: str
    
    # HMM分析结果
    hmm_stage: str
    hmm_confidence: float
    
    # LLM分析结果（如果有）
    llm_stage_agreement: Optional[str]
    llm_suggestion: Optional[str]
    
    # 原始数据摘要
    close_price: float
    analysis_summary: str
    
    # 数据文件路径（存储完整分析结果）
    detail_file: str


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self):
        self.history_dir = HISTORY_DIR
        self.history_dir.mkdir(exist_ok=True)
        self.index_file = self.history_dir / "index.json"
    
    def _generate_record_id(self, stock_code: str) -> str:
        """生成记录ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        code = stock_code.split('.')[0] if '.' in stock_code else stock_code
        return f"{code}_{timestamp}"
    
    def save_analysis(
        self,
        stock_code: str,
        stock_name: str,
        hmm_result: Dict,
        llm_result: Optional[Dict],
        close_price: float,
        full_data: Dict
    ) -> bool:
        """
        保存分析记录
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            hmm_result: HMM分析结果
            llm_result: LLM分析结果（可选）
            close_price: 最新收盘价
            full_data: 完整分析数据（保存到detail文件）
        
        Returns:
            bool: 是否成功
        """
        try:
            record_id = self._generate_record_id(stock_code)
            analysis_time = datetime.now().isoformat()
            
            # 保存详细数据
            detail_file = self.history_dir / f"{record_id}.json"
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, ensure_ascii=False, indent=2)
            
            # 创建索引记录
            record = AnalysisRecord(
                record_id=record_id,
                analysis_time=analysis_time,
                stock_code=stock_code,
                stock_name=stock_name,
                hmm_stage=hmm_result.get('current_stage', '未知'),
                hmm_confidence=hmm_result.get('confidence', 0),
                llm_stage_agreement=llm_result.get('stage_agreement') if llm_result else None,
                llm_suggestion=llm_result.get('suggestion') if llm_result else None,
                close_price=close_price,
                analysis_summary=self._generate_summary(hmm_result, llm_result),
                detail_file=str(detail_file)
            )
            
            # 更新索引
            self._add_to_index(record)
            
            return True
            
        except Exception as e:
            print(f"保存分析记录失败: {e}")
            return False
    
    def _generate_summary(
        self, 
        hmm_result: Dict, 
        llm_result: Optional[Dict]
    ) -> str:
        """生成分析摘要"""
        summary = f"HMM判断: {hmm_result.get('current_stage', '未知')}"
        
        if llm_result and 'stage_agreement' in llm_result:
            summary += f", LLM: {llm_result['stage_agreement']}"
        
        return summary
    
    def _add_to_index(self, record: AnalysisRecord):
        """添加记录到索引"""
        index = self._load_index()
        
        index.append({
            'record_id': record.record_id,
            'analysis_time': record.analysis_time,
            'stock_code': record.stock_code,
            'stock_name': record.stock_name,
            'hmm_stage': record.hmm_stage,
            'hmm_confidence': record.hmm_confidence,
            'llm_stage_agreement': record.llm_stage_agreement,
            'llm_suggestion': record.llm_suggestion,
            'close_price': record.close_price,
            'analysis_summary': record.analysis_summary,
            'detail_file': record.detail_file
        })
        
        # 按时间倒序排列，保留最近100条
        index = sorted(index, key=lambda x: x['analysis_time'], reverse=True)[:100]
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def _load_index(self) -> List[Dict]:
        """加载索引"""
        if not self.index_file.exists():
            return []
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def get_history(
        self, 
        stock_code: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        获取历史记录
        
        Args:
            stock_code: 股票代码（可选，不传返回所有）
            limit: 返回条数
            
        Returns:
            历史记录列表
        """
        index = self._load_index()
        
        if stock_code:
            code = stock_code.split('.')[0] if '.' in stock_code else stock_code
            index = [r for r in index if r['stock_code'].startswith(code)]
        
        return index[:limit]
    
    def get_detail(self, record_id: str) -> Optional[Dict]:
        """获取详细分析数据"""
        try:
            detail_file = self.history_dir / f"{record_id}.json"
            if detail_file.exists():
                with open(detail_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"读取详细数据失败: {e}")
            return None
    
    def delete_record(self, record_id: str) -> bool:
        """删除记录"""
        try:
            # 删除详细文件
            detail_file = self.history_dir / f"{record_id}.json"
            if detail_file.exists():
                detail_file.unlink()
            
            # 更新索引
            index = self._load_index()
            index = [r for r in index if r['record_id'] != record_id]
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"删除记录失败: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        index = self._load_index()
        
        if not index:
            return {
                'total_records': 0,
                'unique_stocks': 0,
                'stage_distribution': {}
            }
        
        # 统计
        unique_stocks = set(r['stock_code'] for r in index)
        stage_counts = {}
        for r in index:
            stage = r['hmm_stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            'total_records': len(index),
            'unique_stocks': len(unique_stocks),
            'stage_distribution': stage_counts
        }
