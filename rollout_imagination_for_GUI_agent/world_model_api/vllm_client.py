#!/usr/bin/env python3

import json
import logging
import time
import os
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai: pip install openai")
    raise

logger = logging.getLogger(__name__)


WORLD_MODEL_PROMPT_TEMPLATE = """You will be given a [Current Screen (OCR+BBOX)] and an [Action List]. Based on the inputs, predict the UI state after the action is executed. Your prediction should include the updated UI state after the action [Post-action Screen (OCR+BBOX)].

[Current Screen (OCR+BBOX)]
{current_screen_ocr}

[Action List]
{action}"""


class WorldModelVLLMClient:
    
    def __init__(
        self,
        base_url: str = "",
        model_name: str = "textual_sketch_world_model",
        api_key: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):

        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 创建 OpenAI 客户端
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        
        # 线程安全的计数器
        self._request_count = 0
        self._request_lock = threading.Lock()
        
        # 用于兼容接口
        self._initialized = True
        
        logger.info(f"WorldModelVLLMClient 初始化完成")
        logger.info(f"  base_url: {base_url}")
        logger.info(f"  model: {model_name}")
        logger.info(f"  max_tokens: {max_tokens}")
        logger.info(f"  temperature: {temperature}")
    
    def _build_prompt(
        self,
        current_screen_ocr: str,
        action: str,
    ) -> str:

        return WORLD_MODEL_PROMPT_TEMPLATE.format(
            current_screen_ocr=current_screen_ocr,
            action=action,
        )
    
    def predict_next_state(
        self,
        current_screen_ocr: str,
        action: str,
        max_retries: int = None,
    ) -> str:
     
        if max_retries is None:
            max_retries = self.max_retries
        
        prompt = self._build_prompt(current_screen_ocr, action)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                with self._request_lock:
                    self._request_count += 1
                
                t0 = time.time()
                
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                
                elapsed = time.time() - t0
                
                predicted_state = response.choices[0].message.content
                
                logger.debug(f"预测完成，耗时: {elapsed:.2f}s")
                
                return predicted_state
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"预测失败 (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
        
        raise RuntimeError(f"预测失败，已重试 {max_retries} 次: {last_error}")
    
    def predict_batch(
        self,
        current_screen_ocr: str,
        actions: List[str],
        max_retries: int = None,
        max_workers: int = None,
    ) -> List[str]:

        if not actions:
            return []
        
        num_actions = len(actions)
        results = [None] * num_actions
        
        if max_workers is None:
            max_workers = min(num_actions, os.cpu_count() or 4, 8)
        
        def predict_one(index: int, action: str) -> Tuple[int, str]:
            try:
                predicted_state = self.predict_next_state(
                    current_screen_ocr=current_screen_ocr,
                    action=action,
                    max_retries=max_retries,
                )
                return index, predicted_state
            except Exception as e:
                logger.error(f"动作 {index} 预测失败: {e}")
                return index, f"[Prediction Failed: {e}]"
        
        t0 = time.time()
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(predict_one, i, action): i 
                for i, action in enumerate(actions)
            }
            
            for future in as_completed(futures):
                try:
                    index, predicted_state = future.result()
                    results[index] = predicted_state
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"获取结果失败 (index={idx}): {e}")
                    results[idx] = f"[Error: {e}]"
        
        elapsed = time.time() - t0
        logger.info(f"批量预测 {num_actions} 个动作完成，总耗时: {elapsed:.2f}s")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        with self._request_lock:
            return {
                "base_url": self.base_url,
                "model_name": self.model_name,
                "total_requests": self._request_count,
                "healthy_endpoints": 1,  # 兼容旧接口
                "total_endpoints": 1,
            }
    
    def health_check(self) -> bool:
        """检查 vLLM 服务是否可用"""
        try:
            # 发送一个简单的请求来检查服务状态
            response = self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"健康检查失败: {e}")
            return False
    
    def format_predicted_state_for_display(self, predicted_state: str) -> str:
        import re
        
        lines = predicted_state.strip().split("\n")
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(
                r"label\s*=\s*(\w+)\s*\|\s*text\s*=\s*'([^']*)'\s*\|\s*bbox\s*=\s*\[([^\]]+)\]",
                line
            )
            if match:
                label = match.group(1)
                text = match.group(2)
                if text:
                    formatted.append(f"  - [{label}] {text}")
            else:
                formatted.append(f"  {line}")
        
        return "\n".join(formatted) if formatted else predicted_state


def create_client(
    base_url: str = None,
    model_name: str = None,
) -> WorldModelVLLMClient:

    if base_url is None:
        base_url = os.environ.get('WORLD_MODEL_VLLM_URL', 'http://localhost:8410/v1')
    
    if model_name is None:
        model_name = os.environ.get('WORLD_MODEL_NAME', 'my-lora')
    
    return WorldModelVLLMClient(base_url=base_url, model_name=model_name)


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8410/v1"
    
    print(f"测试 vLLM 客户端连接: {base_url}")
    client = WorldModelVLLMClient(base_url=base_url)
    
    # 检查服务状态
    if client.health_check():
        print("✓ 服务可用")
        
        print("\n测试单个预测...")
        try:
            result = client.predict_next_state(
                current_screen_ocr="label = text | text='Hello' | bbox=[0, 0, 100, 50]",
                action='action=click, coordinate=[50, 25]'
            )
            print(f"预测成功:\n{result[:500]}...")
        except Exception as e:
            print(f"预测失败: {e}")
        
        print("\n测试批量预测...")
        try:
            actions = [
                'action=click, coordinate=[50, 25]',
                'action=scroll, scroll_direction=down, coordinate=[540, 960]',
                'action=type, text="hello"',
            ]
            results = client.predict_batch(
                current_screen_ocr="label = text | text='Hello' | bbox=[0, 0, 100, 50]",
                actions=actions
            )
            for i, r in enumerate(results):
                print(f"  动作 {i+1}: {r[:100]}...")
        except Exception as e:
            print(f"批量预测失败: {e}")
    else:
        print("✗ 服务不可用")

