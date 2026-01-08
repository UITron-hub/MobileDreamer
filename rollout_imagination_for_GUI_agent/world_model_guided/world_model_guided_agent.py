import sys
import os
import logging
import json
import re
import time
import base64
from io import BytesIO
import numpy as np
from typing import Optional, Union, List, Tuple, Any
from pathlib import Path
from copy import deepcopy
from PIL import Image

ANDROID_WORLD_PATH = "../android_world"
if ANDROID_WORLD_PATH not in sys.path:
    sys.path.insert(0, ANDROID_WORLD_PATH)

from trajectory_rollout.multi_rollout_agent import MultiRolloutAgent, PLANNER_SYS_PROMPT
from trajectory_rollout.client import MtAndroidEnvClient
from trajectory_rollout.agents import OpenAILlmWrapper, LLMEvaluator
from trajectory_rollout.agent_prompts import MOBILE_USE_TOOL_LIST

try:
    from .ocr_module import OCRModule
except ImportError:
    from ocr_module import OCRModule


class WorldModelGuidedAgent(MultiRolloutAgent):

    def __init__(
        self,
        client: MtAndroidEnvClient,
        reasoner_model_name: str,
        actor_model_name: str = "aws.claude-sonnet-4",
        llm_evaluator: Optional[LLMEvaluator] = None,
        wait_after_action_seconds: float = 2.0, 
        save_path: Optional[Union[str, Path]] = None,
        thread_id: int = 0,

        world_model=None,
        ocr_module=None,
        num_candidate_actions: int = 3,
    ):

        super().__init__(
            client=client,
            reasoner_model_name=reasoner_model_name,
            actor_model_name=actor_model_name,
            llm_evaluator=llm_evaluator,
            wait_after_action_seconds=wait_after_action_seconds,
            save_path=save_path,
            thread_id=thread_id,
        )
        
        # 世界模型
        self.world_model = world_model
        
        # OCR 模块
        self.ocr_module = ocr_module or OCRModule()
        
        # 候选动作数量
        self.num_candidate_actions = num_candidate_actions
        
        logging.info(f"WorldModelGuidedAgent 初始化完成")
        logging.info(f"  Reasoner: {reasoner_model_name}")
        logging.info(f"  Actor: {actor_model_name}")
        logging.info(f"  候选动作数: {num_candidate_actions}")
        logging.info(f"  世界模型: {type(world_model).__name__ if world_model else 'None'}")
        logging.info(f"  OCR模块: {type(self.ocr_module).__name__}")
    
    def get_llm_response(self, goal, screenshot_raw, step_data):
        screenshot = self.pre_process_screenshot(screenshot_raw)
        h, w, _ = screenshot.shape
        img_size = (w, h)
        
        if len(self.chat) == 0:
            user_prompt = f"User Instruction: {goal}\nScreenshot: {self.reasoenr_model.image_token}"
            step_data["reasoner_prompt"] = user_prompt
            self.chat = self.reasoenr_model.construct_chat(user_prompt, [screenshot])
        else:
            self.chat.append(self.reasoenr_model.construct_chat(self.reasoenr_model.image_token, [screenshot])[-1])
            self.chat = self.remove_history_contents(self.chat, self.max_img_num_for_llm)
        
        reasoner_output, is_safe, raw_response = self.reasoenr_model.call_llm(self.chat)
        reasoner_output = reasoner_output["content"]
        logging.debug(f"Reasoner output: {reasoner_output}")
        
        if not is_safe:
            return reasoner_output, is_safe, raw_response, step_data
        
        self.chat.append({"role": "assistant", "content": reasoner_output})
        step_data["reasoner_output"] = reasoner_output
        
        operation_match = re.search(r"<operation>(.*)</operation>", reasoner_output, re.DOTALL)
        if operation_match:
            implement = operation_match.group(1).strip()
        else:
            lower_output = reasoner_output.lower()
            if any(kw in lower_output for kw in ['completed', 'complete', 'finished', 'done', 'success', '完成', '成功']):
                logging.info(f"Reasoner 认为任务已完成，返回 status 动作")
                status_action = {
                    "role": "assistant",
                    "content": "Task completed",
                    "tool_calls": [{
                        "id": f"call_status_{int(time.time()*1000)}",
                        "type": "function",
                        "function": {
                            "name": "computer_use",
                            "arguments": json.dumps({"action": "status", "status": "complete"})
                        }
                    }]
                }
                return status_action, True, status_action, step_data
            else:
                logging.error(f"Format error for reasoner output (no <operation> tag): {reasoner_output}")
                raise ValueError(f"Reasoner output missing <operation> tag: {reasoner_output[:200]}")
        
        logging.debug(f"Current step implement: {implement}")
        step_data["action_prompt"] = implement
        
        if len(self.actor_chat) == 0:
            self.actor_chat = self.actor_model.construct_chat(
                self.actor_model.image_token + implement, [screenshot]
            )
            if self.actor_chat[0]["role"] == "system":
                self.actor_chat[0]["content"] += (
                    f"\n\nYour final task is: {goal}\n\n"
                    "The user will take it apart into sub-tasks and provide you each sub-task step by step, "
                    "you need to perform an action to fulfill the sub-task. "
                    "You can freely decide the action details based on the screenshot and final task if necessary."
                )
        else:
            if implement.startswith("Ignore the screenshot"):
                msg = self.actor_model.construct_chat(implement)[-1]
            else:
                msg = self.actor_model.construct_chat(self.actor_model.image_token + implement, [screenshot])[-1]
            self.actor_chat.append(msg)
            self.actor_chat = self.remove_history_contents(
                self.actor_chat, max_imgs_to_keep=3, max_history_turns=5, keep_first_user_content=False
            )
        
        tools = deepcopy(MOBILE_USE_TOOL_LIST)
        tools[0]["function"]["description"] = tools[0]["function"]["description"].replace(
            "<RESOLUTION>", f"{img_size[0]}x{img_size[1]}"
        )
      
        use_world_model = (self.world_model is not None) and (self.num_candidate_actions >= 1)
        if use_world_model:
            candidate_actions = self._generate_candidate_actions(
                goal, implement, screenshot, img_size, tools, step_data
            )

            if len(candidate_actions) > 1:
                best_action, _ = self._select_best_action_with_world_model(
                    goal, screenshot, screenshot_raw, candidate_actions, step_data
                )

                self._update_chat_history(best_action)
                return best_action, True, best_action, step_data

            if len(candidate_actions) == 1:
                action_output = candidate_actions[0]
                try:
                    action_str = self._action_to_string(action_output)
                    current_screen_ocr, predicted_state = self._predict_single_action_outcome(
                        goal=goal,
                        screenshot_raw=screenshot_raw,
                        action_str=action_str,
                        step_data=step_data,
                    )
                    step_data["wm_predictions"] = [predicted_state]
                    step_data["selected_action_idx"] = 0
                    step_data["wm_feedback_text"] = (
                        "World Model Feedback (single-action mode):\n"
                        f"Action: {action_str}\n"
                        f"Predicted Screen After Action:\n{predicted_state}"
                    )

                    self.chat.append({"role": "user", "content": step_data["wm_feedback_text"]})
                except Exception as e:
                    logging.warning(f"单动作世界模型预测失败（忽略并继续 baseline 执行）: {e}")

                self._update_chat_history(action_output)
                return action_output, True, action_output, step_data

            action_output, is_safe, raw_response = self.actor_model.call_llm(self.actor_chat, tools=tools)
        else:  # baseline: 无世界模型或 num_candidate_actions == 0
            action_output, is_safe, raw_response = self.actor_model.call_llm(self.actor_chat, tools=tools)
        
        if not is_safe:
            return action_output, is_safe, raw_response, step_data

        if isinstance(action_output, dict) and action_output.get("content") is not None:
            logging.debug(f"Actor output: {action_output['content']}")
            if action_output.get("reasoning_content") is not None:
                step_data["action_reason"] = action_output["reasoning_content"]
            
            action_output.pop("task_id", None)
            action_output.pop("reasoning_content", None)
            action_output.pop("reasoning_details", None)
            
            self._update_chat_history(action_output)
        
        return action_output, is_safe, raw_response, step_data

    def _predict_single_action_outcome(
        self,
        goal: str,
        screenshot_raw: np.ndarray,
        action_str: str,
        step_data: dict,
    ) -> Tuple[str, str]:

        # OCR 当前屏幕
        current_screen_ocr = self.ocr_module.image_to_text_description(screenshot_raw)

        step_data["current_screen_ocr"] = current_screen_ocr

        predicted_state = self.world_model.predict_next_state(
            current_screen_ocr=current_screen_ocr,
            action=action_str,
        )
        return current_screen_ocr, predicted_state
    
    def _update_chat_history(self, action_output: dict):

        self.actor_chat.append(action_output)
        for tool_call in action_output.get("tool_calls", []):
            self.actor_chat.append({"role": "tool", "tool_call_id": tool_call["id"], "content": "done"})
            self.chat.append({"role": "user", "content": f"Actor executed action: {tool_call['function']['arguments']}"})
    
    def _generate_candidate_actions(
        self,
        goal: str,
        implement: str,
        screenshot: np.ndarray,
        img_size: Tuple[int, int],
        tools: list,
        step_data: dict,
    ) -> List[dict]:

        candidates = []
        
        for i in range(self.num_candidate_actions):
            try:
                temp_chat = deepcopy(self.actor_chat)
                
                if i > 0 and len(candidates) > 0:
                    diversity_hint = f"\n\nNote: Please propose a DIFFERENT action from the previous ones. Previous action was: {self._action_to_string(candidates[-1])[:100]}"
                    if temp_chat[-1].get("content"):
                        if isinstance(temp_chat[-1]["content"], str):
                            temp_chat[-1]["content"] += diversity_hint
                        elif isinstance(temp_chat[-1]["content"], list):
                            temp_chat[-1]["content"].append({"type": "text", "text": diversity_hint})
                
                action_output, is_safe, _ = self.actor_model.call_llm(
                    temp_chat, 
                    tools=tools,
                )
                
                if is_safe and isinstance(action_output, dict) and action_output.get("tool_calls"):
                    action_output.pop("task_id", None)
                    action_output.pop("reasoning_content", None)
                    action_output.pop("reasoning_details", None)
                    
                    action_str = self._action_to_string(action_output)
                    is_duplicate = any(
                        self._action_to_string(c) == action_str 
                        for c in candidates
                    )
                    if not is_duplicate:
                        candidates.append(action_output)
                        
            except Exception as e:
                logging.warning(f"生成候选动作 {i+1} 失败: {e}")
        
        step_data["candidate_actions"] = [
            self._action_to_string(c) for c in candidates
        ]
        logging.info(f"生成了 {len(candidates)} 个候选动作")
        
        return candidates
    
    def _action_to_string(self, action: dict) -> str:
        """将动作字典转换为可读字符串"""
        tool_calls = action.get("tool_calls", [])
        if tool_calls:
            return tool_calls[0]["function"]["arguments"]
        return action.get("content", str(action))
    
    def _select_best_action_with_world_model(
        self,
        goal: str,
        screenshot: np.ndarray,
        screenshot_raw: np.ndarray,
        candidate_actions: List[dict],
        step_data: dict,
    ) -> Tuple[dict, int]:

        try:
            current_screen_ocr = self.ocr_module.image_to_text_description(screenshot_raw)
        except Exception as e:
            logging.error(f"OCR 处理失败: {e}")
            current_screen_ocr = "[OCR Failed]"
        
        step_data["current_screen_ocr"] = current_screen_ocr
        
        predictions = []
        action_strings = [self._action_to_string(a) for a in candidate_actions]
        
        logging.info(f"使用世界模型预测 {len(candidate_actions)} 个动作的结果...")
        
        if hasattr(self.world_model, 'predict_batch') and len(action_strings) > 1:
            try:
                predicted_states = self.world_model.predict_batch(
                    current_screen_ocr=current_screen_ocr,
                    actions=action_strings,
                )
                predictions = predicted_states
            except Exception as e:
                logging.warning(f"批量预测失败，降级为串行: {e}")
                predictions = []
        
        if not predictions:
            for action_str in action_strings:
                try:
                    predicted_state = self.world_model.predict_next_state(
                        current_screen_ocr=current_screen_ocr,
                        action=action_str,
                    )
                    predictions.append(predicted_state)
                except Exception as e:
                    logging.error(f"预测失败: {e}")
                    predictions.append(f"[Prediction Failed: {e}]")
        
        step_data["wm_predictions"] = predictions
        
        selected_idx = self._select_with_reasoner(
            goal, screenshot, current_screen_ocr, action_strings, predictions, step_data
        )
        
        step_data["selected_action_idx"] = selected_idx
        logging.info(f"Reasoner 选择了动作 {selected_idx + 1}")
        
        return candidate_actions[selected_idx], selected_idx
    
    def _select_with_reasoner(
        self,
        goal: str,
        screenshot: np.ndarray,
        current_screen_ocr: str,
        action_strings: List[str],
        predictions: List[str],
        step_data: dict,
    ) -> int:

        predictions_text = ""
        for i, (action, prediction) in enumerate(zip(action_strings, predictions), 1):
            pred_display = prediction[:500] + "..." if len(str(prediction)) > 500 else prediction
            predictions_text += f"\n--- Action {i} ---\n"
            predictions_text += f"Action: {action}\n"
            predictions_text += f"Predicted Screen After Action: {pred_display}\n"
        
        selection_prompt = f"""You are selecting the best action based on World Model predictions.

Task: {goal}

Current sub-task: {step_data.get("action_prompt", "")}

The World Model has predicted what the screen will look like after each candidate action:
{predictions_text}

Based on the CURRENT SCREEN (shown in the image) and these predictions, which action is most likely to make progress toward the task goal?

Reply with ONLY:
<selection>
Action Number: [1-{len(action_strings)}]
Reason: [Brief reason]
</selection>
"""
        
        temp_chat = deepcopy(self.chat)
        selection_msg = self._build_multimodal_message(screenshot, selection_prompt)
        temp_chat.append(selection_msg)
        
        try:
            selection_output, is_safe, _ = self.reasoenr_model.call_llm(temp_chat)
            
            if is_safe and isinstance(selection_output, dict):
                content = selection_output.get("content", "")
                step_data["selection_reason"] = content
                
                match = re.search(r"Action Number:\s*(\d+)", content)
                if match:
                    idx = int(match.group(1)) - 1
                    if 0 <= idx < len(action_strings):
                        return idx
                
                for pattern in [
                    r"<selection>.*?Action Number:\s*(\d+)",
                    r"select.*?action\s*(\d+)",
                    r"Action\s*(\d+)\s+is",
                    r"choose.*?action\s*(\d+)",
                    r"(\d+)", 
                ]:
                    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if match:
                        idx = int(match.group(1)) - 1
                        if 0 <= idx < len(action_strings):
                            return idx
                            
        except Exception as e:
            logging.error(f"Reasoner 选择失败: {e}")
        
        logging.warning("无法解析选择结果，默认选择第一个动作")
        return 0
    
    def _build_multimodal_message(self, screenshot: np.ndarray, text: str) -> dict:
        pil_image = Image.fromarray(screenshot)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        image_url = f"data:image/png;base64,{base64_str}"
        
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}
            ]
        }
    
    @property
    def actor_model(self):
        return self.llm


