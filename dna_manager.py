"""
股票DNA管理模块
保存和加载股票的HMM模型DNA
"""
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from hmm_analyzer import StockDNA
from config import DNA_DIR


class DNAManager:
    """股票DNA管理器"""
    
    def __init__(self):
        self.dna_dir = DNA_DIR
        self.dna_dir.mkdir(exist_ok=True)
    
    def _get_dna_path(self, stock_code: str) -> Path:
        """获取DNA文件路径"""
        # 标准化代码
        code = stock_code.split('.')[0] if '.' in stock_code else stock_code
        return self.dna_dir / f"{code}_dna.json"
    
    def save_dna(self, dna: StockDNA) -> bool:
        """
        保存股票DNA到本地
        
        Args:
            dna: StockDNA对象
            
        Returns:
            bool: 是否成功
        """
        try:
            filepath = self._get_dna_path(dna.stock_code)
            
            # 更新保存时间
            dna.updated_at = datetime.now().isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dna.to_dict(), f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存DNA失败: {e}")
            return False
    
    def load_dna(self, stock_code: str) -> Optional[StockDNA]:
        """
        加载股票DNA
        
        Args:
            stock_code: 股票代码
            
        Returns:
            StockDNA对象，如果不存在返回None
        """
        try:
            filepath = self._get_dna_path(stock_code)
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return StockDNA.from_dict(data)
        except Exception as e:
            print(f"加载DNA失败: {e}")
            return None
    
    def dna_exists(self, stock_code: str) -> bool:
        """检查DNA是否存在"""
        filepath = self._get_dna_path(stock_code)
        return filepath.exists()
    
    def list_all_dna(self) -> Dict[str, dict]:
        """
        列出所有保存的DNA
        
        Returns:
            {stock_code: dna_info}
        """
        dna_list = {}
        
        for filepath in self.dna_dir.glob("*_dna.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stock_code = data.get('stock_code', filepath.stem.replace('_dna', ''))
                dna_list[stock_code] = {
                    'stock_name': data.get('stock_name', ''),
                    'created_at': data.get('created_at', ''),
                    'updated_at': data.get('updated_at', ''),
                    'total_days': data.get('total_days', 0),
                    'stage_distribution': data.get('stage_distribution', {})
                }
            except Exception as e:
                print(f"读取DNA文件失败 {filepath}: {e}")
        
        return dna_list
    
    def delete_dna(self, stock_code: str) -> bool:
        """
        删除股票DNA
        
        Args:
            stock_code: 股票代码
            
        Returns:
            bool: 是否成功
        """
        try:
            filepath = self._get_dna_path(stock_code)
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            print(f"删除DNA失败: {e}")
            return False
    
    def get_dna_summary(self, stock_code: str) -> Optional[dict]:
        """获取DNA摘要信息（不加载完整数据）"""
        dna = self.load_dna(stock_code)
        if dna is None:
            return None
        
        return {
            'stock_code': dna.stock_code,
            'stock_name': dna.stock_name,
            'created_at': dna.created_at,
            'updated_at': dna.updated_at,
            'total_days': dna.total_days,
            'stage_distribution': dna.stage_distribution,
            'stage_templates': list(dna.stage_templates.keys())
        }
