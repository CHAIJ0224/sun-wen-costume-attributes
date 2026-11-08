import argparse
import csv
import json
import os
from pathlib import Path
import random
from datetime import datetime
from pathlib import Path
try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.utils.dataset import CSVCostumeDataset, create_data_loaders
from torch.utils.data import Subset, WeightedRandomSampler
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    # 固定随机种子，尽量让每次训练结果可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size):
    # 训练集加一点增强，验证和测试只做基础缩放。
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_transform, eval_transform



class MultiTaskCostumeClassifier(nn.Module):
    """
    Multi-task costume classifier with shared backbone and task-specific heads.
    Supports 8 simultaneous costume classification tasks.
    """
    
    # Task configuration: (task_name, num_classes)
    TASKS = [
        ('overall', 4),      # 整体
        ('collar', 3),       # 领型
        ('sleeve', 4),       # 袖型
        ('structure', 4),    # 结构
        ('length', 4),       # 衣长 (0-3)
        ('waist', 2),        # 束腰
        ('layer', 3),        # 层次 (0-2)
        ('identity', 5),     # 性别/身份
    ]
    
    def __init__(self, backbone_name='convnext_tiny', pretrained=True):
        super().__init__()
        
        # 共享的 backbone 编码器
        base_model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.backbone = base_model
        self.feat_dim = base_model.num_features
        
        # 每个任务的分类头
        self.heads = nn.ModuleDict()
        for task_name, num_classes in self.TASKS:
            self.heads[task_name] = nn.Linear(self.feat_dim, num_classes)
        
        self.task_names = [t[0] for t in self.TASKS]
    
    def forward(self, x):
        """
        Forward pass returning logits for all tasks.
        
        Args:
            x: input images [B, 3, H, W]
        
        Returns:
            dict of {task_name: logits} where logits are [B, num_classes]
        """
        features = self.backbone(x)  # [B, feat_dim]
        
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.heads[task_name](features)
        
        return outputs
    
    def predict_single_task(self, x, task_name):
        """
        Predict for a single task only (useful for Grad-CAM).
        
        Returns: logits [B, num_classes]
        """
        features = self.backbone(x)
        return self.heads[task_name](features)


class GradCAMTaskWrapper(nn.Module):
    """Wrap a multi-task model so GradCAM sees a single-task tensor output."""

    def __init__(self, model, task_name):
        super().__init__()
        self.model = model
        self.task_name = task_name

    @property
    def backbone(self):
        return self.model.backbone if hasattr(self.model, "backbone") else self.model

    def forward(self, x):
        return self.model.predict_single_task(x, self.task_name)


def create_model(model_name, pretrained=True):
    """
    Create a classification model.
    
    Args:
        model_name: backbone name (e.g., 'convnext_tiny')
        pretrained: whether to use pretrained weights
    
    Returns:
        Model instance
    """
    model = MultiTaskCostumeClassifier(backbone_name=model_name, pretrained=pretrained)

    return model.to(device)


def compute_task_class_weights_from_loader(data_loader, eps: float = 1e-6):
    """
    Compute per-task inverse-frequency class weights from a DataLoader's underlying dataset.

    Returns dict: {task_name: [w0, w1, ...]}
    """
    ds = data_loader.dataset
    # handle Subset wrappers
    if isinstance(ds, Subset):
        base = ds.dataset
        indices = ds.indices
    else:
        base = ds
        # base is CSVCostumeDataset, with records mapping
        indices = list(range(len(base.records)))

    from collections import Counter
    task_counts = {task: Counter() for task, _ in MultiTaskCostumeClassifier.TASKS}
    task_totals = {task: 0 for task, _ in MultiTaskCostumeClassifier.TASKS}

    for idx in indices:
        rec = base.records[idx]
        row = base.df.iloc[rec['row_index']]
        for task, info in MultiTaskCostumeClassifier.TASKS:
            val = int(row[f'label_{task}'])
            if task == 'layer' and val >= 1:
                val = val - 1
            if val >= 0:
                task_counts[task][val] += 1
                task_totals[task] += 1

    class_weights = {}
    for task, info in MultiTaskCostumeClassifier.TASKS:
        num_classes = info[1] if isinstance(info, tuple) else info
        total = task_totals[task]
        counts = [task_counts[task].get(c, 0) for c in range(num_classes)]
        weights = []
        for c in counts:
            if c > 0:
                w = (total / (num_classes * c + eps))
            else:
                w = 1.0
            weights.append(float(w))
        class_weights[task] = weights

    return class_weights


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean', eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float))
        else:
            self.weight = None

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        # gather log probabilities of targets
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.weight is not None:
            at = self.weight[targets]
        else:
            at = 1.0

        loss = -at * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def evaluate(model, data_loader, device_='cuda'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: model instance
        data_loader: DataLoader
        multi_task: if True, expects multi-task labels dict
        device_: device to use
    
    Returns:
        dict with metrics
    """
    model.eval()
    
    # Multi-task evaluation
    all_preds = {task: [] for task, _ in MultiTaskCostumeClassifier.TASKS}
    all_labels = {task: [] for task, _ in MultiTaskCostumeClassifier.TASKS}
    all_confidences = {task: [] for task, _ in MultiTaskCostumeClassifier.TASKS}
        
    with torch.no_grad():
        for imgs, labels_dict in data_loader:
            imgs = imgs.to(device_)
            outputs = model(imgs)
                
            for task_name in model.task_names:
                logits = outputs[task_name]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                confs, _ = torch.max(probs, dim=1)
                confs = confs.cpu().numpy()
                    
                # 获取标签，转移到 CPU 并过滤缺失值（-1）
                batch_labels = labels_dict[task_name].cpu().numpy()
                valid_mask = batch_labels >= 0
                    
                all_preds[task_name].extend(preds[valid_mask])
                all_labels[task_name].extend(batch_labels[valid_mask])
                all_confidences[task_name].extend(confs[valid_mask])
        
    # 计算每个任务的指标
    metrics = {}
    for task_name in model.task_names:
        if len(all_labels[task_name]) > 0:
            acc = accuracy_score(all_labels[task_name], all_preds[task_name])
            macro_f1 = f1_score(all_labels[task_name], all_preds[task_name], 
                                average='macro', zero_division=0)
            avg_conf = np.mean(all_confidences[task_name])
                
            metrics[task_name] = {
                'accuracy': float(acc),
                'macro_f1': float(macro_f1),
                'avg_confidence': float(avg_conf),
                'preds': all_preds[task_name],
                'labels': all_labels[task_name],
                'confidences': all_confidences[task_name],
            }
        else:
            metrics[task_name] = {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'avg_confidence': 0.0,
                'preds': [],
                'labels': [],
                'confidences': [],
            }
        
    return metrics
    


def train(model, train_loader, val_loader, learning_rate, num_epochs, output_dir, model_name,
          task_weights=None, use_focal=True, focal_gamma=2.0, oversample=True):
    """
    Train model with optional multi-task support.
    
    Args:
        task_weights: dict of {task_name: weight} for loss weighting, default is uniform
    """
    # Multi-task training
    if task_weights is None:
        task_weights = {task: 1.0 for task, _ in MultiTaskCostumeClassifier.TASKS}
        
    # 计算每个任务的 class weights
    try:
        class_weights = compute_task_class_weights_from_loader(train_loader)
        print(f"[训练] 计算到 per-task class_weights")
    except Exception as exc:
        print(f"[训练] 无法计算 class_weights, 将使用均等权重: {exc}")
        class_weights = {task: None for task, _ in MultiTaskCostumeClassifier.TASKS}

    # 如果需要过采样，基于样本在各任务上的少数类权重构建样本权重
    if oversample:
        ds = train_loader.dataset
        if isinstance(ds, Subset):
            base = ds.dataset
            indices = ds.indices
        else:
            base = ds
            indices = list(range(len(base.records)))

        sample_weights = []
        for idx in indices:
            rec = base.records[idx]
            row = base.df.iloc[rec['row_index']]
            # compute sample-level weight as max of inverse-freq weights among tasks where label exists
            wvals = []
            for task, _ in MultiTaskCostumeClassifier.TASKS:
                val = int(row[f'label_{task}'])
                if task == 'layer' and val >= 1:
                    val = val - 1
                if val >= 0 and class_weights.get(task) is not None:
                    cw = class_weights[task]
                    if val < len(cw):
                        wvals.append(cw[val])
            sample_weights.append(max(wvals) if len(wvals) > 0 else 1.0)

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        # recreate train_loader with sampler (preserve batch_size, collate_fn, num_workers)
        train_loader = DataLoader(base if not isinstance(ds, Subset) else Subset(base, indices),
                                  batch_size=train_loader.batch_size,
                                  sampler=sampler,
                                  num_workers=getattr(train_loader, 'num_workers', 0),
                                  collate_fn=getattr(train_loader, 'collate_fn', None))
        print(f"[训练] 已启用 WeightedRandomSampler 进行过采样")

    # 构建每个任务的损失函数（使用 FocalLoss 或 CrossEntropyLoss）
    criteria = {}
    for task, num_cls in MultiTaskCostumeClassifier.TASKS:
        if isinstance(num_cls, tuple):
            num_cls = num_cls[1]

        w = None
        if class_weights and class_weights.get(task) is not None:
            try:
                w = torch.tensor(class_weights[task], dtype=torch.float).to(device)
            except Exception:
                w = None

        if use_focal:
            criteria[task] = FocalLoss(gamma=focal_gamma, weight=w, reduction='mean')
        else:
            if w is not None:
                criteria[task] = nn.CrossEntropyLoss(weight=w, reduction='mean')
            else:
                criteria[task] = nn.CrossEntropyLoss(reduction='mean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_f1 = -1.0
    history = []
    best_ckpt_path = os.path.join(output_dir, f"{model_name}_best.pth")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        print(f"[分类] {model_name} 开始第 {epoch + 1}/{num_epochs} 个 epoch")

        for batch_idx, (imgs, labels_dict) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            
            outputs = model(imgs)
            
            # 计算加权多任务损失
            loss = 0.0
            has_valid_loss = False
            
            for task_name in model.task_names:
                logits = outputs[task_name]
                
                # 获取该任务的标签，转移到设备
                batch_labels = labels_dict[task_name].to(device)
                
                # 过滤缺失值（-1）
                valid_mask = batch_labels >= 0
                num_valid = valid_mask.sum().item()
                
                if num_valid > 0:
                    # 只对有效样本计算损失
                    valid_logits = logits[valid_mask]
                    valid_labels = batch_labels[valid_mask]
                    
                    task_loss = criteria[task_name](valid_logits, valid_labels)
                    loss += task_weights[task_name] * task_loss
                    has_valid_loss = True
            
            # 只在有有效的损失时进行反向传播
            if has_valid_loss and loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            if batch_idx == 1 or batch_idx % 10 == 0:
                if has_valid_loss:
                    print(f"--- batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}")
                else:
                    print(f"--- batch {batch_idx}/{len(train_loader)} (no valid labels)")

        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader)
        
        # 以第一个任务的 F1 作为检查点依据
        first_task = model.task_names[0]
        val_f1 = val_metrics[first_task]['macro_f1']
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "val_f1": float(val_f1),
        })

        print(f"--- Epoch [{epoch + 1}/{num_epochs}] TrainLoss={avg_loss:.4f} ValF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
                "epoch": epoch + 1,
            }, best_ckpt_path)

    history_path = os.path.join(output_dir, f"{model_name}_train_log.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "val_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return best_ckpt_path, history_path


def save_metrics_and_reports(metrics, output_dir, model_name, split_name):
    """
    Save evaluation metrics and per-class reports.
    
    Args:
        metrics: dict of metrics (single-task or multi-task)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Multi-task reporting
    summary = {}
    for task_name, task_metrics in metrics.items():
        if len(task_metrics['labels']) == 0:
            continue
        
        # 混淆矩阵
        cm = confusion_matrix(task_metrics['labels'], task_metrics['preds'])
        cm_path = os.path.join(output_dir, f"{model_name}_{split_name}_{task_name}_confusion_matrix.csv")
        np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
        
        # 分类报告
        report = classification_report(
            task_metrics['labels'],
            task_metrics['preds'],
            output_dict=True,
            zero_division=0,
        )
        report_path = os.path.join(output_dir, f"{model_name}_{split_name}_{task_name}_per_class_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        summary[task_name] = {
            "accuracy": task_metrics['accuracy'],
            "macro_f1": task_metrics['macro_f1'],
            "avg_confidence": task_metrics['avg_confidence'],
            "confusion_matrix": cm_path,
            "per_class_report": report_path,
        }
    
    # 保存总结
    summary_path = os.path.join(output_dir, f"{model_name}_{split_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return {"summary": summary_path, "tasks": summary}


def select_target_layer(model):
    """
    Select appropriate target layer for Grad-CAM.
    """
    # MultiTaskCostumeClassifier 的 backbone
    if hasattr(model, 'backbone'):
        base_model = model.backbone
    else:
        base_model = model
    
    if hasattr(base_model, "stages"):
        return base_model.stages[-1]
    if hasattr(base_model, "layer4"):
        return base_model.layer4[-1]
    if hasattr(base_model, "blocks"):
        return base_model.blocks[-1]
    return list(base_model.children())[-1]


def visualize_gradcam(model, img_path, image_size=224, output_path="gradcam.jpg", task_name='overall'):
    """
    生成单张图的 Grad-CAM，用于检查模型到底看了哪里。
    
    Args:
        task_name: 对于多任务模型，指定要可视化的任务
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    cam_model = GradCAMTaskWrapper(model, task_name)
    target_layer = select_target_layer(cam_model)
    cam = GradCAM(model=cam_model, target_layers=[target_layer])

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img.resize((image_size, image_size))).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


def generate_gradcam_samples(model, dataset_dir, output_dir, image_size=224, max_images=8):
    """
    批量导出少量样本热力图，够论文展示即可，不必全量输出。
    """
    os.makedirs(output_dir, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    saved = 0

    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            if Path(file_name).suffix.lower() not in image_exts:
                continue
            img_path = os.path.join(root, file_name)
            
            # 为每个任务生成 Grad-CAM
            for task_name in model.task_names:
                out_name = f"cam_{saved:03d}_{task_name}_{Path(file_name).stem}.jpg"
                out_path = os.path.join(output_dir, out_name)
                try:
                    visualize_gradcam(model, img_path, image_size=image_size, 
                                    output_path=out_path, task_name=task_name)
                except Exception as exc:
                    print(f"Skip Grad-CAM for {img_path} (task {task_name}): {exc}")
            
            saved += 1
            if saved >= max_images:
                return saved
    
    return saved


'''baseline模型的分类水平还没有达到完全可靠的程度，暂时注释掉批量推理功能
def _resolve_prediction_image_paths(image_dir):
    """收集待推理图片，兼容 person_det 的单文件命名和 clothes_roi 的前缀文件命名。"""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    resolved = []

    for file_name in sorted(os.listdir(image_dir)):
        if Path(file_name).suffix.lower() not in image_exts:
            continue

        stem = Path(file_name).stem
        if stem.isdigit():
            image_id = int(stem)
            roi_index = None
        else:
            prefix = stem.split("_", 1)[0]
            if not prefix.isdigit():
                continue
            image_id = int(prefix)
            roi_index = stem[len(prefix) + 1:] if len(stem) > len(prefix) + 1 else None

        resolved.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": os.path.join(image_dir, file_name),
                "roi_index": roi_index,
            }
        )

    return resolved

def predict_batch_multi_task(model, image_dir, output_csv, confidence_threshold=0.5):
    """
    批量推理：对 person_det 或 clothes_roi 文件夹中的所有图片进行多任务预标注。
    
    Args:
        model: MultiTaskCostumeClassifier 模型
        image_dir: person_det 文件夹路径
        output_csv: 输出 CSV 路径
        confidence_threshold: 高于此阈值的预测标记为"已自动标注"，否则标记为"需要人工复核"
    
    Returns:
        DataFrame with predictions
    """
    
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # 收集所有图片
    image_records = _resolve_prediction_image_paths(image_dir)
    print(f"[推理] 找到 {len(image_records)} 张图片")
    
    results = []
    
    with torch.no_grad():
        for idx, record in enumerate(image_records):
            if (idx + 1) % 50 == 0:
                print(f"[推理] 已处理 {idx + 1}/{len(image_records)} 张图片")
            
            img_path = record["image_path"]
            image_id = record["image_id"]
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                outputs = model(img_tensor)
                
                row = {
                    'image_path': image_id,
                    'source_file': record['file_name'],
                }
                if record['roi_index'] is not None:
                    row['roi_index'] = record['roi_index']
                # 对每个任务取 top-1 和置信度
                for task_name in model.task_names:
                    logits = outputs[task_name][0]  # [num_classes]
                    probs = torch.softmax(logits, dim=0)
                    conf, pred = torch.max(probs, dim=0)
                    
                    row[f'label_{task_name}'] = int(pred.cpu().item())
                    row[f'conf_{task_name}'] = float(conf.cpu().item())
                
                # 计算平均置信度作为复核判断依据
                avg_conf = np.mean([row[f'conf_{task}'] for task in model.task_names])
                row['avg_confidence'] = avg_conf
                row['needs_review'] = avg_conf < confidence_threshold
                
                results.append(row)
                
            except Exception as e:
                print(f"[推理] Error processing {record['file_name']}: {e}")
                # 添加空行标记失败
                row = {
                    'image_path': image_id,
                    'source_file': record['file_name'],
                }
                if record['roi_index'] is not None:
                    row['roi_index'] = record['roi_index']
                for task_name in model.task_names:
                    row[f'label_{task_name}'] = -1
                    row[f'conf_{task_name}'] = 0.0
                row['avg_confidence'] = 0.0
                row['needs_review'] = True
                results.append(row)
    
    # 保存为 CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[推理] 预标注结果已保存: {output_csv}")
    
    # 统计需要复核的样本数
    need_review_count = df_results['needs_review'].sum()
    auto_labeled_count = len(df_results) - need_review_count
    print(f"[推理] 自动标注: {auto_labeled_count}, 需要复核: {need_review_count}")
    
    return df_results
'''

def build_parser():
    parser = argparse.ArgumentParser(description="Train/evaluate multi-task costume classifier or batch predict")
    
    # 数据源选择
    parser.add_argument("--mode", type=str, default="csv", 
                       choices=["csv", "predict"],
                       help="Data source mode: 'csv' for CSV-driven multi-task, 'imagefolder' for traditional, 'predict' for batch inference")
    
    # 数据集模式参数
    parser.add_argument("--csv_path", type=str, default="dataset\\labels.csv",
                       help="Path to normalized CSV annotations (for csv mode)")
    parser.add_argument("--image_dir", type=str, default="dataset\\person_crops",
                       help="Path to image directory (for csv mode, e.g. person_det or clothes_roi_yolov8)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train set ratio (for csv mode)")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation set ratio (for csv mode)")
    
    # 模型参数
    parser.add_argument("--image_size", type=int, default=224, help="Image size for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", help="Model architecture to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/classification", help="Output directory")
    parser.add_argument("--gradcam_dir", type=str, default=None, help="Grad-CAM output directory")
    parser.add_argument("--gradcam_max_images", type=int, default=8, help="Max Grad-CAM images to export")
    
    # 推理模式参数
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                       help="Path to model checkpoint (for predict mode)")
    parser.add_argument("--predict_dir", type=str, default="dataset\person_det",
                       help="Directory with images to predict (for predict mode, e.g. person_det or clothes_roi_yolov8)")
    parser.add_argument("--predict_output", type=str, default="dataset\predictions.csv",
                       help="Output CSV path for predictions (for predict mode)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                       help="Confidence threshold for auto-labeling (for predict mode)")
    
    return parser


def main():
    # 主流程：训练 -> 载入最佳模型 -> 验证/测试 -> 导出报告和可视化。
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    """
    # 推理模式
    if args.mode == "predict":
        print("[推理] 进入批量预标注模式")
        if not args.checkpoint_path:
            print("[ERROR] 推理模式需要指定 --checkpoint_path")
            return
        if not os.path.exists(args.checkpoint_path):
            print(f"[ERROR] 检查点不存在: {args.checkpoint_path}")
            return
        if not os.path.exists(args.predict_dir):
            print(f"[ERROR] 图像目录不存在: {args.predict_dir}")
            return
        
        # 载入模型
        model = MultiTaskCostumeClassifier(
            backbone_name=args.model_name,
            pretrained=True,
        )
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        print(f"[推理] 已载入检查点: {args.checkpoint_path}")
        
        # 批量推理
        df_predictions = predict_batch_multi_task(
            model=model,
            image_dir=args.predict_dir,
            output_csv=args.predict_output,
            confidence_threshold=args.confidence_threshold,
        )
        print("[推理] 批量预标注完成")
        return
    """

    # 训练模式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir_name = Path(args.image_dir).name or args.image_dir.replace("\\", "_").replace("/", "_")
    run_output_dir = os.path.join(args.output_dir, f"run-{image_dir_name}-{args.model_name}-{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"[分类] 使用 CSV 多任务模式")
    if not os.path.exists(args.csv_path):
        print(f"[ERROR] CSV 文件不存在: {args.csv_path}")
        return
    if not os.path.exists(args.image_dir):
        print(f"[ERROR] 图像目录不存在: {args.image_dir}")
        return
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        class_names = None
    except Exception as e:
        print(f"[ERROR] 加载 CSV 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    all_models = ["convnext_tiny", "resnet18", "efficientnet_b0", "efficientnet_b2"]
    if args.model_name == "all":
        model_list = all_models
    elif args.model_name in all_models:
        model_list = [args.model_name]
    else:
        print(f"Invalid model name: {args.model_name}. Valid options are: {all_models} or 'all'.")
        return

    run_summary = {
        "seed": args.seed,
        "device": device,
        "mode": args.mode,
        "models": {},
    }
    run_summary["class_names"] = class_names

    for model_name in model_list:
        print(f"[分类] 开始训练和评估 {model_name}...")
        model = create_model(model_name=model_name)
        print(f"[分类] 已创建模型：{model_name}")
        
        best_ckpt_path, history_path = train(
            model,
            train_loader,
            val_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            output_dir=run_output_dir,
            model_name=model_name,
        )

        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_metrics = evaluate(model, val_loader)
        test_metrics = evaluate(model, test_loader)

        val_report_paths = save_metrics_and_reports(val_metrics, run_output_dir, model_name, "val")
        test_report_paths = save_metrics_and_reports(test_metrics, run_output_dir, model_name, "test")

        first_task = list(test_metrics.keys())[0]
        test_acc = test_metrics[first_task]['accuracy']
        test_f1 = test_metrics[first_task]['macro_f1']
        print(f"[分类] {model_name} 完成。TestAcc={test_acc:.4f}, TestF1={test_f1:.4f}")

        run_summary["models"][model_name] = {
            "checkpoint": best_ckpt_path,
            "train_log": history_path,
            "val_reports": val_report_paths,
            "test_reports": test_report_paths,
        }

    summary_path = os.path.join(run_output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    print("Run summary saved to", summary_path)
    print("[分类] 全部流程完成")
    generate_gradcam_samples(model=model, dataset_dir="dataset/person_det", output_dir="outputs/gradcam_samples", image_size=224, max_images=8)


if __name__ == "__main__":
    main()
    