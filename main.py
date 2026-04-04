# main.py
import os
import time
from config import INPUT_DIR
from core.clip_classifier import CLIPClassifier
from core.file_manager import FileManager


def get_image_paths(folder_path: str) -> list:
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    ]


def main():
    print("=== Image Auto Organizer (CLIP-based) ===")
    print("分类类别: 人物, 风景, 动物, 其他")

    # 1. 获取图片
    image_paths = get_image_paths(INPUT_DIR)
    if not image_paths:
        print("❌ input 文件夹中未找到图片")
        return
    print(f"✅ 找到 {len(image_paths)} 张图片")

    # 2. 初始化CLIP分类器
    print("正在加载 CLIP 模型...")
    start_time = time.time()
    classifier = CLIPClassifier()

    # 3. 批量预测
    print("开始分类...")
    predictions, valid_paths = classifier.batch_predict(image_paths)

    if not predictions:
        print("❌ 没有图片成功处理")
        return

    # 4. 整理文件
    manager = FileManager()
    manager.organize_by_category(valid_paths, predictions)

    # 5. 性能统计
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f}秒")
    print(f"🚀 平均速度: {len(predictions) / elapsed:.1f} 张/秒")


if __name__ == "__main__":
    main()