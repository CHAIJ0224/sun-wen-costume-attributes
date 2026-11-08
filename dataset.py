"""
CSVCostumeDataset: CSV-driven multi-task dataset for costume classification.

Reads normalized annotation CSV (class_result_clean.csv) and loads images from person_det folder.
Supports 8 costume classification tasks simultaneously.
"""

import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def collate_multitask_batch(batch):
    """
    自定义 collate 函数处理多任务标签字典。
    将一个批次的 (image, labels_dict) 转换为 (images_batch, labels_dict_batch)
    """
    images = []
    labels_by_task = {}
    
    for img, label_dict in batch:
        images.append(img)
        for task_name, label in label_dict.items():
            if task_name not in labels_by_task:
                labels_by_task[task_name] = []
            labels_by_task[task_name].append(label)
    
    # 堆叠图像
    images_batch = torch.stack(images)
    
    # 堆叠每个任务的标签成张量
    labels_dict_batch = {
        task_name: torch.tensor(labels, dtype=torch.long)
        for task_name, labels in labels_by_task.items()
    }
    
    return images_batch, labels_dict_batch


class CSVCostumeDataset(Dataset):
    """
    Multi-task costume classification dataset driven by CSV annotations.
    
    Args:
        csv_path: Path to normalized CSV file (class_result_clean.csv)
        image_dir: Path to person_det folder containing images
        transform: torchvision.transforms.Compose for image preprocessing
        
    Returns per sample:
        (image_tensor, label_dict) where label_dict contains 8 task labels
        Labels with value -1 are treated as missing (masked during training)
    """
    
    # Task names and their label ranges
    TASK_INFO = {
        'overall': {'idx': 0, 'num_classes': 4, 'name': '整体'},          # 0-3
        'collar': {'idx': 1, 'num_classes': 3, 'name': '领型'},           # 0-2
        'sleeve': {'idx': 2, 'num_classes': 4, 'name': '袖型'},           # 0-3
        'structure': {'idx': 3, 'num_classes': 4, 'name': '结构'},        # 0-3
        'length': {'idx': 4, 'num_classes': 4, 'name': '衣长'},           # 0-3
        'waist': {'idx': 5, 'num_classes': 2, 'name': '束腰'},            # 0-1
        'layer': {'idx': 6, 'num_classes': 3, 'name': '层次'},            # 1-3 in CSV, remapped to 0-2 for training
        'identity': {'idx': 7, 'num_classes': 5, 'name': '性别/身份'},    # 0-4
    }
    
    TASK_NAMES = list(TASK_INFO.keys())
    
    def __init__(self, csv_path, image_dir, transform=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()
        self.image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        # 读取 CSV
        self.df = pd.read_csv(csv_path)
        print(f"[Dataset] Loaded {len(self.df)} annotations from {csv_path}")
        
        # 验证必要列
        required_cols = ['image_path'] + [f'label_{task}' for task in self.TASK_NAMES]
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        # 统计缺失标签
        missing_stats = {}
        for task in self.TASK_NAMES:
            col = f'label_{task}'
            missing_count = (self.df[col] == -1).sum()
            missing_stats[task] = missing_count
        
        print(f"[Dataset] Missing label statistics:")
        for task, count in missing_stats.items():
            print(f"  {task}: {count}/{len(self.df)}")

        # 为每个原始 image_id 建立实际文件映射：
        # - person_det: 通常是 image_id.jpg
        # - clothes_roi: 可能有 image_id_0.jpg, image_id_1.jpg 等多个裁切
        self.records = []
        self.group_to_indices = {}
        self.group_order = []
        self.missing_images = []

        for row_index, row in self.df.iterrows():
            image_id = int(row['image_path'])
            matched_paths = self._resolve_image_paths(image_id)

            if not matched_paths:
                self.missing_images.append(image_id)
                continue

            group_id = str(image_id)
            if group_id not in self.group_to_indices:
                self.group_to_indices[group_id] = []
                self.group_order.append(group_id)

            for img_path in matched_paths:
                sample_index = len(self.records)
                self.records.append({
                    'row_index': row_index,
                    'image_id': image_id,
                    'image_path': img_path,
                    'group_id': group_id,
                })
                self.group_to_indices[group_id].append(sample_index)

        if self.missing_images:
            preview = self.missing_images[:10]
            print(
                f"[Dataset] Warning: {len(self.missing_images)} image_id(s) have no matching files in {image_dir}; "
                f"examples: {preview}"
            )

        print(f"[Dataset] Resolved {len(self.records)} sample files from {len(self.df)} annotation rows")

    def _resolve_image_paths(self, image_id):
        """查找与 image_id 对应的实际图像文件，支持单文件和 ROI 前缀文件。"""
        exact_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if os.path.exists(exact_path):
            return [exact_path]

        prefix = f"{image_id}_"
        matched = []
        for file_name in sorted(os.listdir(self.image_dir)):
            suffix = Path(file_name).suffix.lower()
            if suffix not in self.image_exts:
                continue
            stem = Path(file_name).stem
            if stem == str(image_id) or stem.startswith(prefix):
                matched.append(os.path.join(self.image_dir, file_name))

        return matched
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        row = self.df.iloc[record['row_index']]
        
        # 读取图像（支持 person_det 的单文件，也支持 clothes_roi 的多文件）
        img_path = record['image_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 读取标签
        labels = {}
        for task in self.TASK_NAMES:
            label_value = int(row[f'label_{task}'])
            if task == 'layer' and label_value >= 1:
                label_value = label_value - 1
            labels[task] = label_value
        
        return image, labels
    
    @classmethod
    def get_num_classes(cls, task):
        """Return number of classes for a specific task."""
        return cls.TASK_INFO[task]['num_classes']
    
    @classmethod
    def get_all_task_info(cls):
        """Return information about all tasks."""
        return cls.TASK_INFO
    
    @classmethod
    def split_dataset(cls, csv_path, image_dir, train_ratio=0.7, val_ratio=0.15, 
                      transform=None, seed=42):
        """
        Split dataset into train/val/test sets.
        
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        dataset = cls(csv_path, image_dir, transform)
        
        groups = list(dataset.group_order)
        n_groups = len(groups)
        n_train = int(n_groups * train_ratio)
        n_val = int(n_groups * val_ratio)
        n_test = n_groups - n_train - n_val

        # 固定随机种子保证可重现
        rng = np.random.default_rng(seed)
        shuffled_groups = np.array(groups, dtype=object)
        rng.shuffle(shuffled_groups)

        train_groups = set(shuffled_groups[:n_train])
        val_groups = set(shuffled_groups[n_train:n_train+n_val])
        test_groups = set(shuffled_groups[n_train+n_val:])

        train_indices = []
        val_indices = []
        test_indices = []

        for group_id in groups:
            indices = dataset.group_to_indices[group_id]
            if group_id in train_groups:
                train_indices.extend(indices)
            elif group_id in val_groups:
                val_indices.extend(indices)
            else:
                test_indices.extend(indices)

        print(
            f"[Dataset] Split groups: train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}; "
            f"samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )
        
        # 创建子集
        from torch.utils.data import Subset
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)
        
        return train_set, val_set, test_set


def create_data_loaders(csv_path, image_dir, image_size=224, batch_size=16, 
                        train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create train/val/test data loaders from CSV.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 定义转换
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # 分割数据集
    train_set, val_set, test_set = CSVCostumeDataset.split_dataset(
        csv_path, image_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        transform=None,  # 先不用转换，后面单独应用
        seed=seed
    )
    
    # 对训练集应用训练转换，对验证/测试集应用评估转换
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = eval_transform
    test_set.dataset.transform = eval_transform
    
    # 创建 DataLoader - 使用自定义 collate 函数处理多任务标签字典
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=0, collate_fn=collate_multitask_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=0, collate_fn=collate_multitask_batch)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=0, collate_fn=collate_multitask_batch)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试
    csv_path = "class_result_clean.csv"
    image_dir = "person_det"
    
    if os.path.exists(csv_path) and os.path.exists(image_dir):
        train_loader, val_loader, test_loader = create_data_loaders(
            csv_path, image_dir, batch_size=4
        )
        
        print(f"\nTesting dataset...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images shape={images.shape}")
            print(f"  Labels: {labels}")
            if batch_idx >= 0:
                break
    else:
        print(f"CSV or image directory not found")
