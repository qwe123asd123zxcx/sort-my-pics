# core/file_manager.py
import os
import shutil
from config import OUTPUT_DIR, FILE_OPERATION_MODE
from core.clip_classifier import CLASS_DIR_NAMES


class FileManager:
    def __init__(self, base_output_dir=OUTPUT_DIR):
        self.base_output_dir = base_output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)

        # 提前创建所有可能的分类文件夹
        for category_cn in CLASS_DIR_NAMES.values():
            category_dir = os.path.join(self.base_output_dir, category_cn)
            os.makedirs(category_dir, exist_ok=True)

    def organize_by_category(self, image_paths: list, predictions: list):
        """根据CLIP预测结果整理文件"""
        for path, pred in zip(image_paths, predictions):
            # 获取中文文件夹名
            target_dir = os.path.join(
                self.base_output_dir,
                pred["category_cn"]
            )

            filename = os.path.basename(path)
            target_path = os.path.join(target_dir, filename)

            if FILE_OPERATION_MODE == "move":
                shutil.move(path, target_path)
            else:  # copy
                shutil.copy2(path, target_path)

            print(f"Moved {filename} -> {pred['category_cn']} (confidence: {pred['confidence']:.2f})")

        print(f"\nAll files organized into {self.base_output_dir}")
        self.print_summary(predictions)

    def print_summary(self, predictions):
        """打印分类统计"""
        from collections import Counter
        counter = Counter([p["category_cn"] for p in predictions])

        print("\n📊 分类统计:")
        for category, count in counter.items():
            print(f"  {category}: {count} 张")