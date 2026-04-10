# core/adaptive_clip_classifier.py
import torch
import torch.nn as nn
from cn_clip.clip import load_from_name
from PIL import Image
import os
from config import (
    CLIP_MODEL_NAME,
    MODEL_DIR,
    CLASS_DIR_NAMES
)
from core.adaptive_head import AdaptiveHead


class AdaptiveCLIPClassifier:
    def __init__(self, device=None, head_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载 CLIP（冻结）
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model, self.preprocess = load_from_name(
            CLIP_MODEL_NAME,
            device=self.device,
            download_root='./models',
            use_modelscope=True
        )
        self.model.eval()

        # 冻结 CLIP
        for p in self.model.parameters():
            p.requires_grad = False

        # 2. 分类头
        self.head = AdaptiveHead().to(self.device)

        if head_path and os.path.exists(head_path):
            self.head.load_state_dict(
                torch.load(head_path, map_location=self.device)
            )
            print(f"✅ 加载分类头: {head_path}")

        self.head.train()

        # 3. 训练组件
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=1e-3
        )

    # ---------- 推理 ----------
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(img)
            logits = self.head(feat)

        pred = logits.argmax(dim=-1).item()
        conf = torch.softmax(logits, dim=-1).max().item()

        return {
            "category": list(CLASS_DIR_NAMES.keys())[pred],
            "category_cn": list(CLASS_DIR_NAMES.values())[pred],
            "confidence": conf
        }

    # ---------- 反向传播 ----------
    def update(self, image_path, true_label_idx):
        """
        true_label_idx: int (0~3)
        """
        image = Image.open(image_path).convert("RGB")
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        label = torch.tensor([true_label_idx]).to(self.device)

        self.optimizer.zero_grad()

        feat = self.model.encode_image(img)
        logits = self.head(feat)

        loss = self.criterion(logits, label)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ---------- 保存 ----------
    def save_head(self, path):
        torch.save(self.head.state_dict(), path)
        print(f"✅ 分类头已保存至 {path}")