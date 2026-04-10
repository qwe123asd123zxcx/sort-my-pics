# ImageSorter: 基于 CLIP 的智能图片分类工具

![version](https://img.shields.io/badge/version-0.2.0-green)
![version](https://img.shields.io/badge/python-3.8+-blue)

这是一个基于 OpenAI CLIP (中文版) 实现的图片自动分类脚本。
能够根据图片内容，将其自动归类到 **人物、风景、动物、其他** 四个文件夹中。

## 功能特性
- ✅ 支持中文语义理解，自动识别图片内容。
- ✅ 无需手动标注，零样本分类。
- ✅ 自动创建分类文件夹并移动文件。

## 环境要求
- Python 3.8+
- PyTorch
- 推荐使用虚拟环境

## 安装与运行

1. **克隆仓库**
git clone https://github.com/qwe123asd123zxcx/sort-my-pics.git
cd imagesorter

2. **安装依赖**
pip install -r requirements.txt

3. **准备数据**
将需要分类的图片放入 `data/input` 文件夹。

4. **运行主程序**
python main.py

5. **查看结果**
分类后的图片将保存在 `data/output` 目录下，分为“人物”、“风景”、“动物”、“其他”四个子目录。

## 项目结构
- `main.py`: 程序入口
- `core/`: 核心逻辑 (CLIP分类器、文件管理)
- `data/input`: 输入图片目录
- `data/output`: 输出分类目录
- `models/`: 模型缓存目录 (自动生成)

## 技术栈
- Python
- PyTorch
- Chinese-CLIP
- CLIP

---
*Powered by Chinese-CLIP*