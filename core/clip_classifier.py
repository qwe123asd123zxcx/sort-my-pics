# core/clip_classifier.py
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, tokenize
from PIL import Image
import os
from config import CLIP_MODEL_NAME, CLASS_LABELS, CLASS_DIR_NAMES,MODEL_DIR


class CLIPClassifier:
    def __init__(self, device=None):
        # 1. 加载模型
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 确保模型目录存在
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"加载中文CLIP模型: {CLIP_MODEL_NAME}")
        self.model, self.preprocess = load_from_name(
            CLIP_MODEL_NAME,
            device=self.device,
            download_root='./models',
            use_modelscope = True
        )
        self.model.eval()
        print(f"模型加载完成，使用设备: {self.device}")

        # 2. 准备提示词文本
        # 用多个同义词增强分类准确性
        self.prompts = self._prepare_prompts()

        # 3. 预编码文本特征（提高效率）
        with torch.no_grad():
            self.text_features = self._encode_texts(self.prompts)

    def _prepare_prompts(self):
        """准备所有类别的提示词"""
        all_prompts = []
        self.label_to_idx = {}  # 记录每个提示词属于哪个类别

        idx = 0
        for category, prompts in CLASS_LABELS.items():
            for prompt in prompts:
                all_prompts.append(prompt)
                self.label_to_idx[idx] = category
                idx += 1

        return all_prompts

    def _encode_texts(self, texts):
        """编码所有文本提示词"""
        text_tokens = tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    """编码中文文本"""
    def predict_single(self, image_path):
        """对单张图片分类"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarity = (image_features @ self.text_features.T).squeeze(0)
            best_idx = similarity.argmax().item()
            best_score = similarity[best_idx].item()
            category = self.label_to_idx[best_idx]

            return {
                "category": category,
                "category_cn": CLASS_DIR_NAMES[category],
                "confidence": best_score,
                "similarities": similarity.cpu().numpy()
            }

        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")
            return None

    def predict(self, image_path):
        """
        统一对外接口
        """
        result = self.predict_single(image_path)
        if result is None:
            raise ValueError(f"Failed to predict {image_path}")
        return result

    def batch_predict(self, image_paths):
        """批量预测"""
        results = []
        valid_paths = []

        for path in image_paths:
            result = self.predict_single(path)
            if result is not None:
                results.append(result)
                valid_paths.append(path)
            else:
                print(f"Skipped: {path}")

        return results, valid_paths

    def get_all_categories(self):
        """获取所有类别（用于创建文件夹）"""
        return list(CLASS_DIR_NAMES.keys())