# utils/image_utils.py
import cv2
import numpy as np
import os

def load_image_opencv(image_path: str) -> np.ndarray:
    """
    使用 OpenCV 读取图片
    返回: HWC, BGR, uint8
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    return img


def resize_and_normalize_for_clip(
    img_bgr: np.ndarray,
    target_size=(224, 224)
) -> np.ndarray:
    """
    将 OpenCV 图片转换为 CLIP 所需格式
    返回: CHW, RGB, float32, normalized
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # resize
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # HWC -> CHW
    img_chw = np.transpose(img_resized, (2, 0, 1))

    # normalize (ImageNet 均值方差)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_norm = (img_chw / 255.0 - mean) / std

    return img_norm.astype(np.float32)