"""
OpenAI 兼容的 LLM 分析模块
支持 OpenAI、Azure、Ollama、Kimi、Claude、Gemini 等任何 OpenAI 兼容的 API
"""
import json
from typing import Dict, List, Optional
import requests


class OpenAIAnalyzer:
    """OpenAI 兼容 API 分析器
    
    支持任何 OpenAI 兼容的 API，包括：
    - OpenAI (https://api.openai.com/v1)
    - Azure OpenAI
    - Kimi/Moonshot (https://api.moonshot.cn/v1)
    - Ollama (http://localhost:11434/v1)
    - Claude (通过 OpenAI 兼容接口)
    - Gemini (通过 OpenAI 兼容接口)
    - 以及其他任何兼容 OpenAI API 格式的服务
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://api.openai.com/v1", 
        model: str = "gpt-3.5-turbo",
        custom_headers: Optional[Dict[str, str]] = None
    ):
        """
        初始化 OpenAI 兼容 API 分析器
        
        Args:
            api_key: API 密钥
            base_url: API 基础地址
            model: 模型名称
            custom_headers: 自定义请求头（可选）
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.custom_headers = custom_headers or {}
        
        # 检测是否为 Azure OpenAI（URL 中包含 openai.azure.com）
        self.is_azure = 'openai.azure.com' in base_url or 'azure.com' in base_url
        
        # 构建 API URL
        if self.is_azure:
            # Azure OpenAI 的 URL 格式不同
            self.api_url = f"{self.base_url}/chat/completions?api-version=2024-02-01"
        else:
            self.api_url = f"{self.base_url}/chat/completions"
        
        # 构建请求头
        self.headers = self._build_headers()
    
    def _build_headers(self) -> Dict[str, str]:
        """构建 API 请求头"""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.is_azure:
            # Azure OpenAI 使用 api-key 头
            headers["api-key"] = self.api_key
        else:
            # 标准 OpenAI 兼容 API 使用 Authorization 头
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 添加自定义 headers
        headers.update(self.custom_headers)
        
        return headers
    
    def analyze_stock(
        self,
        stock_code: str,
        stock_name: str,
        hmm_result: Dict,
        recent_data_summary: Dict,
        cycle_prediction: Dict = None
    ) -> Dict:
        """
        使用 LLM 分析股票
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            hmm_result: HMM分析结果
            recent_data_summary: 近期数据摘要
            cycle_prediction: 周期预测结果（可选）
            
        Returns:
            LLM分析结果
        """
        # 构建提示词
        prompt = self._build_prompt(
            stock_code, stock_name, hmm_result, recent_data_summary, cycle_prediction
        )
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是专业的股票技术分析专家，擅长识别股票的主力操作阶段。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return self._parse_response(content, stock_code)
            else:
                return {
                    'stock_code': stock_code,
                    'error': f'API请求失败: {response.status_code}',
                    'raw_response': response.text
                }
                
        except Exception as e:
            return {
                'stock_code': stock_code,
                'error': str(e)
            }
    
    def _build_prompt(
        self,
        stock_code: str,
        stock_name: str,
        hmm_result: Dict,
        recent_data_summary: Dict,
        cycle_prediction: Dict = None
    ) -> str:
        """构建分析提示词"""
        
        prompt = f"""请分析股票 {stock_code} {stock_name} 的当前状态。

【HMM模型分析结果】
- 当前阶段: {hmm_result.get('current_stage', '未知')}
- 置信度: {hmm_result.get('confidence', 0):.2%}
- 状态稳定性: {hmm_result.get('stability', 0):.2%}
- 各阶段概率:
"""
        
        # 添加各阶段概率
        all_probs = hmm_result.get('all_stage_probs', {})
        for stage, prob in all_probs.items():
            prompt += f"  - {stage}: {prob:.2%}\n"
        
        # 添加特征分析
        features = hmm_result.get('feature_analysis', {})
        prompt += "\n【关键特征】\n"
        for name, value in features.items():
            prompt += f"- {name}: {value:.4f}\n"
        
        # 添加近期数据摘要
        prompt += f"""
【近期数据摘要】
- 最新收盘价: {recent_data_summary.get('close', 'N/A')}
- 20日涨跌幅: {recent_data_summary.get('return_20d', 'N/A')}
- 20日均量变化: {recent_data_summary.get('volume_change', 'N/A')}
- 当前价格位置: {recent_data_summary.get('price_position', 'N/A')}
"""
        
        # 添加周期预测信息（如果有）
        if cycle_prediction:
            prompt += f"""
【周期预测信息】
- 当前周期: {cycle_prediction.get('current_stage', '未知')}（已持续{cycle_prediction.get('days_in_cycle', 0)}天）
- 周期进度: {cycle_prediction.get('cycle_progress', 0):.1%}
- 预计周期结束: {cycle_prediction.get('cycle_estimated_end', '未知')}
- 周期结束目标价格区间: {cycle_prediction.get('price_target_low', 0):.2f} - {cycle_prediction.get('price_target_high', 0):.2f}
- 未来5天预测: {cycle_prediction.get('pred_5d_stage', '未知')}，价格{cycle_prediction.get('pred_5d_price_low', 0):.2f}-{cycle_prediction.get('pred_5d_price_high', 0):.2f}
- 未来20天预测: {cycle_prediction.get('pred_20d_stage', '未知')}，价格{cycle_prediction.get('pred_20d_price_low', 0):.2f}-{cycle_prediction.get('pred_20d_price_high', 0):.2f}
"""
        
        prompt += """
你是一位专业的量化分析师，拥有10年以上股票投资研究经验。请基于上述数据，以客观、严谨、专业的角度进行分析。

分析要求：
1. 保持客观中立，不要给出过于乐观或讨好的评价
2. 重点关注数据支持的结论，避免主观臆测
3. 风险提示要具体，不要泛泛而谈
4. 如果数据不足以支持某个判断，请明确指出

请提供以下分析（以JSON格式返回）:
{
  "stage_agreement": "是否同意HMM判断（同意/部分同意/不同意），并简要说明理由",
  "stage_reasoning": "基于技术指标和量价关系的阶段分析，指出支持或反对当前判断的证据",
  "risk_assessment": "风险评级（高/中/低），说明主要风险点",
  "key_signals": ["关键信号1（说明看涨/看跌理由）", "关键信号2"],
  "suggestion": "操作建议（观望/关注/谨慎/回避），说明适用条件",
  "confidence": 0-1之间的数值,
  "cycle_assessment": "对周期预测的评估：判断是否合理，指出可能的偏差",
  "price_outlook": "价格展望（短期1-2周、中期1-3个月），给出价格区间和触发条件",
  "additional_insights": "其他观察：如量价背离、异常波动、与大盘/板块的关系等"
}

注意：
- 如果当前处于高位派发期，请明确提示风险，不要暗示还有上涨空间
- 如果处于低位吸筹期，请说明可能的筑底时间，不要急于建议买入
- 价格预测请给出触发条件（如突破/跌破某个价位）
"""
        return prompt
    
    def _parse_response(self, content: str, stock_code: str) -> Dict:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0].strip()
            else:
                json_str = content
            
            result = json.loads(json_str)
            result['stock_code'] = stock_code
            result['raw_content'] = content
            return result
            
        except json.JSONDecodeError:
            # 如果解析失败，返回原始内容
            return {
                'stock_code': stock_code,
                'raw_content': content,
                'error': '无法解析JSON响应'
            }
    
    def batch_analyze(
        self,
        analysis_inputs: List[Dict]
    ) -> List[Dict]:
        """
        批量分析多只股票
        
        Args:
            analysis_inputs: 每个元素是包含stock_code, stock_name, hmm_result, recent_data的字典
            
        Returns:
            分析结果列表
        """
        results = []
        for input_data in analysis_inputs:
            result = self.analyze_stock(
                input_data['stock_code'],
                input_data['stock_name'],
                input_data['hmm_result'],
                input_data['recent_data']
            )
            results.append(result)
        return results
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                },
                timeout=15
            )
            return response.status_code == 200
        except Exception as e:
            print(f"API连接测试失败: {e}")
            return False
    
    @staticmethod
    def get_preset_configs() -> Dict[str, Dict[str, str]]:
        """
        获取预设配置列表
        
        Returns:
            预设配置字典，包含各种常见 OpenAI 兼容服务的配置
        """
        return {
            "OpenAI": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo"
            },
            "Kimi (Moonshot)": {
                "base_url": "https://api.moonshot.cn/v1",
                "model": "moonshot-v1-8k"
            },
            "Ollama (本地)": {
                "base_url": "http://localhost:11434/v1",
                "model": "llama2"
            },
            "Claude (OpenAI兼容)": {
                "base_url": "https://api.anthropic.com/v1",
                "model": "claude-3-sonnet-20240229"
            },
            "Gemini (OpenAI兼容)": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": "gemini-1.5-flash"
            },
            "DeepSeek": {
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat"
            },
            "SiliconFlow": {
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct"
            },
            "Azure OpenAI": {
                "base_url": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
                "model": "gpt-35-turbo"
            },
            "Groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.1-8b-instant"
            },
            "Cerebras": {
                "base_url": "https://api.cerebras.ai/v1",
                "model": "llama3.1-8b"
            },
            "Together AI": {
                "base_url": "https://api.together.xyz/v1",
                "model": "meta-llama/Llama-3.1-8B-Instruct-Turbo"
            },
            "Fireworks": {
                "base_url": "https://api.fireworks.ai/inference/v1",
                "model": "accounts/fireworks/models/llama-v3p1-8b-instruct"
            },
            "LocalAI": {
                "base_url": "http://localhost:8080/v1",
                "model": "gpt-3.5-turbo"
            },
            "vLLM": {
                "base_url": "http://localhost:8000/v1",
                "model": "meta-llama/Llama-2-7b-chat-hf"
            }
        }
