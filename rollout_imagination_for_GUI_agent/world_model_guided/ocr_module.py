"""
OCR Module - 将Android截图转化为世界模型可理解的文本描述
基于 PaddleOCR-VL 实现
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
from PIL import Image

# PaddleOCR 配置
PADDLE_OCR_CONFIG = {
    "layout_detection_model_name": "PP-DocLayoutV2",
    "layout_detection_model_dir": "PP-DocLayoutV2",
    "vl_rec_model_name": "PaddleOCR-VL-0.9B",
    "vl_rec_model_dir": "PaddleOCR-VL",
}


class OCRModule:
    """
    OCR模块：将Android截图转换为世界模型可理解的文本描述
    
    输出格式示例:
    label = text | text='Home' | bbox=[35, 175, 347, 233]
    label = text | text='Settings' | bbox=[100, 200, 300, 250]
    """
    
    def __init__(
        self,
        layout_detection_model_name: str = PADDLE_OCR_CONFIG["layout_detection_model_name"],
        layout_detection_model_dir: str = PADDLE_OCR_CONFIG["layout_detection_model_dir"],
        vl_rec_model_name: str = PADDLE_OCR_CONFIG["vl_rec_model_name"],
        vl_rec_model_dir: str = PADDLE_OCR_CONFIG["vl_rec_model_dir"],
        lazy_init: bool = True,
    ):

        self.layout_detection_model_name = layout_detection_model_name
        self.layout_detection_model_dir = os.path.expanduser(layout_detection_model_dir)
        self.vl_rec_model_name = vl_rec_model_name
        self.vl_rec_model_dir = os.path.expanduser(vl_rec_model_dir)
        
        self.pipeline = None
        self._initialized = False
        
        if not lazy_init:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        if self._initialized:
            return
            
        try:
            from paddleocr import PaddleOCRVL
            
            logging.info("正在初始化PaddleOCR模型...")
            self.pipeline = PaddleOCRVL(
                vl_rec_model_name=self.vl_rec_model_name,
                vl_rec_model_dir=self.vl_rec_model_dir,
                layout_detection_model_name=self.layout_detection_model_name,
                layout_detection_model_dir=self.layout_detection_model_dir,
            )
            self._initialized = True
            logging.info("PaddleOCR模型初始化完成!")
        except ImportError as e:
            logging.error(f"无法导入PaddleOCR: {e}")
            logging.error("请确保已安装 paddleocr 包")
            raise
        except Exception as e:
            logging.error(f"初始化PaddleOCR时出错: {e}")
            raise
    
    def image_to_text_description(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        return_raw: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        将图像转换为世界模型格式的文本描述
        
        Args:
            image: 输入图像 (numpy数组、路径或PIL Image)
            return_raw: 是否返回原始OCR结果
            
        Returns:
            文本描述字符串，格式如:
            label = text | text='Home' | bbox=[35, 175, 347, 233]
            label = text | text='Settings' | bbox=[100, 200, 300, 250]
        """
        if not self._initialized:
            self._initialize_pipeline()

        temp_path = None
        if isinstance(image, np.ndarray):
            fd, temp_path = tempfile.mkstemp(prefix="ocr_temp_", suffix=".png")
            os.close(fd)
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        elif isinstance(image, Image.Image):
            fd, temp_path = tempfile.mkstemp(prefix="ocr_temp_", suffix=".png")
            os.close(fd)
            image.save(temp_path)
            image_path = temp_path
        elif isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        
        # 执行OCR
        try:
            output = self.pipeline.predict(image_path)
            
            # 解析结果
            ocr_elements = []
            raw_results = []
            
            for res in output:
                # 获取识别结果
                if hasattr(res, 'rec_polys') and res.rec_polys is not None:
                    for i, (poly, text) in enumerate(zip(res.rec_polys, res.rec_texts)):
                        if text.strip():  # 忽略空文本
                            # 计算bbox (取多边形的外接矩形)
                            x_coords = [p[0] for p in poly]
                            y_coords = [p[1] for p in poly]
                            bbox = [
                                int(min(x_coords)),
                                int(min(y_coords)),
                                int(max(x_coords)),
                                int(max(y_coords))
                            ]
                            
                            element = {
                                "label": "text",
                                "text": text.strip(),
                                "bbox": bbox
                            }
                            ocr_elements.append(element)
                            raw_results.append({
                                "poly": poly,
                                "text": text,
                                "bbox": bbox
                            })
                
                # 获取布局检测结果
                if hasattr(res, 'layout_det_res') and res.layout_det_res is not None:
                    layout_res = res.layout_det_res
                    if hasattr(layout_res, 'boxes') and layout_res.boxes is not None:
                        for box, label in zip(layout_res.boxes, layout_res.labels):
                            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                            
                            # 查找该区域内的文本
                            text_in_region = self._find_text_in_region(bbox, ocr_elements)
                            
                            element = {
                                "label": label if label else "unknown",
                                "text": text_in_region,
                                "bbox": bbox
                            }
                            # 避免重复添加纯文本区域
                            if label not in ["text", "paragraph"] or not text_in_region:
                                ocr_elements.append(element)
            
            if return_raw:
                return {
                    "text_description": self._format_elements(ocr_elements),
                    "elements": ocr_elements,
                    "raw_results": raw_results
                }
            
            return self._format_elements(ocr_elements)
            
        except Exception as e:
            logging.error(f"OCR处理时出错: {e}")
            if return_raw:
                return {"text_description": "", "elements": [], "raw_results": []}
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    def _find_text_in_region(
        self, 
        region_bbox: List[int], 
        elements: List[Dict]
    ) -> str:
        """查找某区域内的文本内容"""
        texts = []
        rx1, ry1, rx2, ry2 = region_bbox
        
        for elem in elements:
            if elem["label"] == "text":
                ex1, ey1, ex2, ey2 = elem["bbox"]
                # 检查是否在区域内
                if (ex1 >= rx1 and ey1 >= ry1 and ex2 <= rx2 and ey2 <= ry2):
                    texts.append(elem["text"])
        
        return "\n".join(texts) if texts else ""
    
    def _format_elements(self, elements: List[Dict]) -> str:
        """
        将OCR元素格式化为世界模型的文本描述格式
        
        格式: label = {label} | text='{text}' | bbox=[x1, y1, x2, y2]
        """
        lines = []
        for elem in elements:
            text = elem.get("text", "").replace("'", "\\'").replace("\n", "\\n")
            bbox = elem.get("bbox", [0, 0, 0, 0])
            label = elem.get("label", "text")
            
            line = f"label = {label} | text='{text}' | bbox={bbox}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def format_for_world_model(
        self,
        current_screen_ocr: str,
        action_list: List[str],
    ) -> str:
        prompt = f"""You will be given a [Current Screen (OCR+BBOX)] and an [Action List]. Based on the inputs, predict the UI state after the action is executed. Your prediction should include the updated UI state after the action [Post-action Screen (OCR+BBOX)].

[Current Screen (OCR+BBOX)]
{current_screen_ocr}

[Action List]
"""
        for i, action in enumerate(action_list, 1):
            prompt += f"{i}. {action}\n"
        
        return prompt.strip()

