import os
import time
from config import INPUT_DIR, MODEL_DIR
from core.clip_classifier import CLIPClassifier
from core.adaptive_clip_classifier import AdaptiveCLIPClassifier
from core.file_manager import FileManager

MODE = "adaptive"  # "zero_shot" or "adaptive"
HEAD_PATH = os.path.join(MODEL_DIR, "classifier_head.pt")


def get_image_paths(folder_path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    ]


def main():
    print("=== Image Auto Organizer ===")
    print(f"运行模式: {'深度学习' if MODE == 'adaptive' else 'Zero-Shot'}")

    image_paths = get_image_paths(INPUT_DIR)
    if not image_paths:
        print("❌ input 文件夹为空")
        return

    print(f"✅ 找到 {len(image_paths)} 张图片")

    # ✅ 统一初始化
    if MODE == "adaptive":
        classifier = AdaptiveCLIPClassifier(
            head_path=HEAD_PATH if os.path.exists(HEAD_PATH) else None
        )
    else:
        classifier = CLIPClassifier()

    predictions = []
    valid_paths = []

    start_time = time.time()

    for img in image_paths:
        result = classifier.predict(img)
        print(
            f"📷 {os.path.basename(img)} "
            f"→ {result['category_cn']} "
            f"(置信度: {result['confidence']:.2f})"
        )

        if MODE == "adaptive":
            fb = input("是否正确？(y/n): ").strip().lower()
            if fb == "n":
                print("0=人物 1=风景 2=动物 3=其他")
                true_label = int(input("输入编号: "))
                loss = classifier.update(img, true_label)
                print(f"✅ 模型已更新，loss={loss:.4f}")

        predictions.append(result)
        valid_paths.append(img)

    FileManager().organize_by_category(valid_paths, predictions)

    if MODE == "adaptive":
        classifier.save_head(HEAD_PATH)

    print(f"\n⏱️ 总耗时: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()