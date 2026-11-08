# 红楼梦服饰识别项目

基于人物检测、服饰分割和多任务分类的服饰属性识别项目。项目支持批量处理图片、导出中间结果、训练分类模型和生成预测结果。

## 运行环境

建议使用 Conda 环境：

```bash
conda activate hongloumeng
pip install -r requirements.txt
```

## 运行流程

### 1. 人物检测

从原始图片中提取人物裁剪图，默认输入 `dataset/images`，输出 `dataset/person_crops`。

```bash
python src/tasks/detector.py --input_folder dataset/images --output_folder dataset/person_crops --model_path yolov8x.pt --save_vis
```

### 2. 服饰区域分割

对人物裁剪图提取服饰 ROI，默认可选 `yolov8` 或 `mask2former`。

```bash
python src/tasks/segmenter.py --input_folder dataset/person_crops --output_folder dataset/clothes_roi --segmenter yolov8 --model_path yolov8x-seg.pt --save_vis
```

### 3. 分类训练

基于 `dataset/result.csv` 和对应图像目录训练多任务分类模型。

```bash
python src/tasks/classifier.py --mode csv --csv_path dataset/result.csv --image_dir dataset/person_crops --model_name convnext_tiny
```

### 4. 批量预测

使用已训练好的模型对新图片批量预标注。

```bash
# 输入：单人物裁剪图；模型：convnext_tiny
python -m src.tasks.classifier --image_dir dataset/person_crops --model_name convnext_tiny
# 输入：基于yolov8分割的服饰ROI图；模型：convnext_tiny
python -m src.tasks.classifier --image_dir dataset/clothes_roi_yolov8 --model_name convnext_tiny
# 输入：基于mask2former分割的服饰ROI图；模型：convnext_tiny
python -m src.tasks.classifier --image_dir dataset/clothes_roi_mask2former --model_name convnext_tiny
# 输入：单人物裁剪图；模型：efficientnet_b2
python -m src.tasks.classifier --image_dir dataset/person_crops --model_name efficientnet_b2
# 输入：基于yolov8分割的服饰ROI图；模型：efficientnet_b2
python -m src.tasks.classifier --image_dir dataset/clothes_roi_yolov8 --model_name efficientnet_b2
# 输入：基于mask2former分割的服饰ROI图；模型：efficientnet_b2
python -m src.tasks.classifier --image_dir dataset/clothes_roi_mask2former --model_name efficientnet_b2
```

## 目录说明

- `src/tasks/`：核心任务脚本，包括人物检测、服饰分割和分类推理/训练。
- `src/utils/`：数据集处理。
- `dataset/images/`：原始输入图片。
- `dataset/person_crops/`：人物检测后的裁剪结果及统计信息。其下的 `vis` 文件夹显示了各个图片的裁切框。
- `dataset/clothes_roi_yolov8/`、`dataset/clothes_roi_mask2former/`：服饰分割后的 ROI 结果及统计信息。其下的 `vis` 文件夹显示了各个图片的裁切框。
- `dataset/labels.csv`：规范化后的标注数据。
- `outputs/classification/`：分类任务运行时产生的训练、评估、预测结果和模型文件。其下的6个 `run-*-*` 文件夹是6种任务配置的各自运行结果，`run_summary.json` 汇总了6种任务配置在测试集上的结果，其余文件是报告中所用到的图片素材。
- `output/gradcam_samples/`：报告中用到的Grad-CAM 示例。
- 根目录下的 `yolov8x.pt`、`yolov8x-seg.pt`、`yolov8n-pose.pt`：检测和分割任务用到的检测/分割/姿态相关模型权重。

## 备注

- 默认路径已写入脚本参数，直接运行也可以生成对应输出。
- 如需更换输入目录、模型或阈值，可通过命令行参数修改、或在源码中修改。
