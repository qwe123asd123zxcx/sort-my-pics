# config.py
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入输出路径
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# 模型配置
#MODEL_NAME = "resnet50"
#IMG_SIZE = (224, 224)

# 聚类配置
#NUM_CLUSTERS = 5
#RANDOM_STATE = 42

# 文件操作模式: 'copy' 或 'move'
FILE_OPERATION_MODE = "copy"

# CLIP 模型选择
CLIP_MODEL_NAME = "ViT-B-16"
MODEL_DIR = os.path.join(BASE_DIR, "models", "clip")  # 模型保存到项目目录

# 分类标签（用中文描述）
CLASS_LABELS = {
    "people": ["一个人", "多人合影", "人像特写", "自拍", "人脸"],
    "scenery": ["风景", "山水", "自然风光", "城市风光", "建筑"],
    "animals": ["猫", "狗", "动物", "宠物", "野生动物"],
    "other": ["其他", "物品", "物体", "抽象", "未知"]
}

# 简化版文件夹命名（用于创建目录）
CLASS_DIR_NAMES = {
    "people": "人物",
    "scenery": "风景",
    "animals": "动物",
    "other": "其他"
}