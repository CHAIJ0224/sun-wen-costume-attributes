from ultralytics import YOLO
import os
import shutil
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image


def iter_image_paths(input_folder):
    # 只处理图片文件，跳过目录和其他无关文件。
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        if os.path.isfile(img_path) and Path(img_name).suffix.lower() in exts:
            yield img_name, img_path


def clamp_bbox(x1, y1, x2, y2, width, height):
    """将边界框限制在图像范围内，并保证坐标顺序有效。"""
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2

def compute_pose_aware_roi_top(image, pose_model, pose_margin=10, bbox=None):
    """
    使用姿态检测 (YOLOv8-pose) 自动找到肩膀位置，作为服饰 ROI 的上边界。
    
    COCO 格式的关键点：id=5 是左肩，id=6 是右肩。肩膀位置是衣服的自然上界。
    
    Args:
        image: 输入图像 (H, W, 3)
        pose_model: 加载好的 YOLOv8-pose 模型
        pose_margin: 肩膀 y 坐标下方的边距（像素），用于包含肩膀线
        bbox: 检测框 (x1, y1, x2, y2)，可选，用于范围约束
    
    Returns:
        roi_top_y: ROI 上边界 y 坐标
    """
    try:
        # 运行 pose 检测
        pose_results = pose_model(image, verbose=False)
        
        if len(pose_results) == 0 or pose_results[0].keypoints is None:
            # 无检测结果，回退
            return None
        
        keypoints = pose_results[0].keypoints.data[0]  # shape: (17, 3) for COCO
        
        # COCO 格式关键点 ID：5=left_shoulder, 6=right_shoulder
        left_shoulder = keypoints[5]   # (x, y, conf)
        right_shoulder = keypoints[6]  # (x, y, conf)
        
        # 只有置信度都足够高的关键点才可信
        CONFIDENCE_THRESHOLD = 0.3
        if left_shoulder[2] < CONFIDENCE_THRESHOLD or right_shoulder[2] < CONFIDENCE_THRESHOLD:
            return None
        
        # 取两肩中较低的 y 值
        shoulder_y = min(left_shoulder[1], right_shoulder[1])
        
        roi_top_y = int(shoulder_y) - pose_margin
        
        # 如果指定了 bbox，约束到该范围内
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi_top_y = max(int(y1), min(roi_top_y, int(y2) - 1))
        
        return roi_top_y
    
    except Exception as e:
        if False:  # 改为 True 以启用调试
            print(f"[Pose-aware ROI] Error: {e}")
        return None


class SegmentationModel(ABC):
    """分割模型的统一接口"""
    @abstractmethod
    def segment(self, image):
        """返回 masks 和 boxes，格式统一"""
        pass


class YOLOv8SegModel(SegmentationModel):
    """YOLOv8 分割模型包装器"""
    def __init__(self, model_path="yolov8x-seg.pt"):
        self.model = YOLO(model_path)
        self.name = "yolov8-seg"
        # COCO 格式中 person 类是 ID 0
        self.person_id = 0

    def segment(self, image):
        # 返回 (masks, boxes) 元组，格式与原实现一致，仅保留 person 类实例
        results = self.model(image, verbose=False)
        if len(results) == 0 or results[0].masks is None:
            return None, None
        
        all_masks = results[0].masks.data  # (N, H, W)
        all_boxes = results[0].boxes  # 检测框
        
        # 过滤只保留 person 类
        person_indices = []
        for i, box in enumerate(all_boxes):
            if int(box.cls[0]) == self.person_id:
                person_indices.append(i)
        
        if len(person_indices) == 0:
            return None, None
        
        # 只保留 person 类的 masks 和 boxes
        masks = all_masks[person_indices]
        boxes = [all_boxes[i] for i in person_indices]
        return masks, boxes


class Mask2FormerModel(SegmentationModel):
    """Mask2Former 分割模型包装器"""
    def __init__(self, model_name="facebook/mask2former-swin-large-coco-panoptic"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.name = "mask2former"
        
        # 获取 id2label 映射来识别 person 类
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
            self.id2label = self.model.config.id2label
        else:
            # 默认 COCO 映射：person 是 class_id 0
            self.id2label = {0: "person"}
        
        # 查找 person 的 class_id
        self.person_id = None
        for class_id, class_name in self.id2label.items():
            if class_name.lower() == "person":
                self.person_id = class_id
                break

    def segment(self, image):
        # Mask2Former 返回通用的实例分割结果，仅保留 person 实例
        try:
            # 转换为 PIL Image 格式
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = image

            # 预处理
            inputs = self.processor(images=image_pil, return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 后处理：使用官方 processor 生成实例分割结果
            target_sizes = [(image.shape[0], image.shape[1])]
            processed = self.processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
            )[0]

            segmentation = processed["segmentation"]
            segments_info = processed["segments_info"]

            masks = []
            boxes = []

            if segmentation is None or len(segments_info) == 0:
                return None, None

            # 按每个实例的 mask 重新生成二值掩码和包围框（仅保留 person 类）
            for segment in segments_info:
                # 关键点：只保留 person 类的实例
                label_id = segment.get("label_id", -1)
                if self.person_id is not None and label_id != self.person_id:
                    continue
                
                instance_mask = (segmentation == segment["id"]).cpu().numpy().astype(np.uint8)
                if instance_mask.sum() == 0:
                    continue

                masks.append(instance_mask)

                y_indices, x_indices = np.where(instance_mask > 0)
                if len(y_indices) > 0:
                    x1, y1 = x_indices.min(), y_indices.min()
                    x2, y2 = x_indices.max(), y_indices.max()
                    boxes.append([x1, y1, x2, y2])

            if len(masks) == 0 or len(boxes) == 0:
                return None, None

            # 返回格式与 YOLOv8 一致
            masks_tensor = torch.stack([torch.from_numpy(m) for m in masks]) if masks else None
            return masks_tensor, boxes

        except Exception as e:
            print(f"Mask2Former segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def create_segmenter(segmenter_name, model_path=None):
    """工厂函数：根据名字创建分割器"""
    if segmenter_name == "yolov8":
        return YOLOv8SegModel(model_path or "yolov8x-seg.pt")
    elif segmenter_name == "mask2former":
        return Mask2FormerModel()
    else:
        raise ValueError(f"Unknown segmenter: {segmenter_name}. Choose from: yolov8, mask2former")


def extract_clothes_roi(
    input_folder,
    output_folder,
    segmenter_name="yolov8",
    model_path=None,
    roi_top_ratio=0.25,
    min_size=20,
    save_visualization=False,
    visualization_dir=None,
    pose_margin=10,
):
    # 输出目录用于保存服饰 ROI 和统计结果。
    os.makedirs(output_folder, exist_ok=True)
    failed_dir = os.path.join(output_folder, "failed_images")
    warn_log_path = os.path.join(output_folder, "segmentation_warnings.log")
    os.makedirs(failed_dir, exist_ok=True)
    if save_visualization:
        if visualization_dir is None:
            visualization_dir = os.path.join(output_folder, "vis")
        os.makedirs(visualization_dir, exist_ok=True)

    # 创建分割器实例
    segmenter = create_segmenter(segmenter_name, model_path)
    
    # 如果启用 pose-aware 模式，加载 YOLOv8-pose 模型
    pose_model = None
    try:
        pose_model = YOLO("yolov8n-pose.pt")
        print(f"[分割] 已加载 YOLOv8-pose 模型")
    except Exception as e:
        raise ValueError(f"[分割] Warning: 加载 YOLOv8-pose 失败: {e}")

    print(
        f"[分割] 开始处理：segmenter={segmenter.name}, input={input_folder}, output={output_folder}, "
        f"roi_top_ratio={roi_top_ratio}, min_size={min_size}"
    )

    total_images = 0
    unreadable_images = 0
    images_with_masks = 0
    total_instances = 0
    saved_rois = 0
    skipped_small_roi = 0
    skipped_empty_roi = 0
    segmenter_failures = 0
    progress_step = 10

    for img_name, img_path in iter_image_paths(input_folder):
        total_images += 1
        img = cv2.imread(img_path)
        if img is None:
            unreadable_images += 1
            continue

        if total_images == 1 or total_images % progress_step == 0:
            print(f"[分割] 已处理 {total_images} 张图片，当前保存 ROI {saved_rois} 张")

        # 使用统一的分割接口
        masks, boxes = segmenter.segment(img)

        mask_found_in_image = False

        if masks is None or boxes is None:
            # 分割失败：用 pose 检测框作为兜底
            segmenter_failures += 1

            fallback_boxes = None
            fallback_reason = "no fallback boxes"
            try:
                pose_results = pose_model(img, verbose=False)
                if len(pose_results) > 0 and getattr(pose_results[0], "boxes", None) is not None:
                    fallback_boxes = pose_results[0].boxes
            except Exception as e:
                fallback_reason = f"pose detection error: {e}"

            if fallback_boxes is None or len(fallback_boxes) == 0:
                try:
                    shutil.copy2(img_path, os.path.join(failed_dir, img_name))
                except Exception:
                    pass
                with open(warn_log_path, "a", encoding="utf-8") as warn_log:
                    warn_log.write(f"{img_name}\t{fallback_reason}\n")
                continue

            boxes_list = fallback_boxes
            total_instances += len(boxes_list)
            vis_img = img.copy() if save_visualization else None

            selected_idx = None
            try:
                box_areas = []
                for idx, box in enumerate(boxes_list):
                    if hasattr(box, "xyxy"):
                        x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    elif isinstance(box, (list, tuple)) and len(box) == 4:
                        x1b, y1b, x2b, y2b = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    else:
                        x1b = y1b = x2b = y2b = 0
                    box_areas.append(max(0, x2b - x1b) * max(0, y2b - y1b))
                if len(box_areas) > 0:
                    selected_idx = int(np.argmax(box_areas))
            except Exception:
                selected_idx = 0

            if selected_idx is None:
                try:
                    shutil.copy2(img_path, os.path.join(failed_dir, img_name))
                except Exception:
                    pass
                with open(warn_log_path, "a", encoding="utf-8") as warn_log:
                    warn_log.write(f"{img_name}\tinvalid fallback boxes\n")
                continue

            box = boxes_list[selected_idx]
            if hasattr(box, "xyxy"):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                try:
                    shutil.copy2(img_path, os.path.join(failed_dir, img_name))
                except Exception:
                    pass
                with open(warn_log_path, "a", encoding="utf-8") as warn_log:
                    warn_log.write(f"{img_name}\tunsupported fallback box format\n")
                continue

            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img.shape[1], img.shape[0])
            clothes_y1 = compute_pose_aware_roi_top(
                img, pose_model, pose_margin=pose_margin,
                bbox=(x1, y1, x2, y2)
            )

            if clothes_y1 is None:
                clothes_y1 = int(y1 + roi_top_ratio * (y2 - y1))

            clothes_y1 = max(y1, min(clothes_y1, y2 - 1))
            clothes_roi = img[clothes_y1:y2, x1:x2]

            if clothes_roi.size == 0:
                try:
                    shutil.copy2(img_path, os.path.join(failed_dir, img_name))
                except Exception:
                    pass
                with open(warn_log_path, "a", encoding="utf-8") as warn_log:
                    warn_log.write(f"{img_name}\tempty fallback roi\n")
                continue

            if clothes_roi.shape[0] < min_size or clothes_roi.shape[1] < min_size:
                try:
                    shutil.copy2(img_path, os.path.join(failed_dir, img_name))
                except Exception:
                    pass
                with open(warn_log_path, "a", encoding="utf-8") as warn_log:
                    warn_log.write(f"{img_name}\tsmall fallback roi\n")
                continue

            save_name = f"{os.path.splitext(img_name)[0]}.jpg"
            save_path = os.path.join(output_folder, save_name)

            ok = cv2.imwrite(save_path, clothes_roi)
            if ok:
                saved_rois += 1
                if save_visualization:
                    cv2.rectangle(vis_img, (x1, clothes_y1), (x2, y2), (255, 140, 0), 2)

            if save_visualization and vis_img is not None:
                vis_name = f"{os.path.splitext(img_name)[0]}_seg.jpg"
                cv2.imwrite(os.path.join(visualization_dir, vis_name), vis_img)

            continue
        else:
            mask_found_in_image = True

            # 对于 YOLOv8，boxes 是对象；对于 Mask2Former，是列表
            boxes_list = boxes
            total_instances += len(boxes_list)

            vis_img = img.copy() if save_visualization else None

            # 如果有多个实例，仅保留掩码面积最大的一个（单人图场景假设）
            selected_idx = None
            try:
                mask_areas = []
                for idx in range(len(boxes_list)):
                    m = masks[idx]
                    if hasattr(m, 'cpu'):
                        m_np = m.cpu().numpy()
                    else:
                        m_np = m if isinstance(m, np.ndarray) else np.array(m)
                    # resize 到图片大小以便公平比较面积
                    m_resized = cv2.resize(m_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    m_bin = (m_resized > 0.5).astype(np.uint8)
                    mask_areas.append(int(m_bin.sum()))
                if len(mask_areas) > 0:
                    selected_idx = int(np.argmax(mask_areas))
            except Exception:
                # 若在面积计算时出现问题，回退到处理第一个实例
                selected_idx = 0

            if selected_idx is None:
                # 无有效实例，跳过
                continue

            # 只处理选中的实例
            i = selected_idx
            box = boxes_list[i]

            # 统一处理 box 格式
            if hasattr(box, 'xyxy'):  # YOLOv8 format
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            elif isinstance(box, (list, tuple)) and len(box) == 4:  # Mask2Former format
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                continue

            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img.shape[1], img.shape[0])

            # 处理掩码格式
            if isinstance(masks, np.ndarray):
                mask = masks[i]
            else:
                mask = masks[i].cpu().numpy() if hasattr(masks[i], 'cpu') else masks[i]

            # 先拿到人物掩码，再裁出下半身区域作为服饰 ROI。
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)

            person_region = img * mask[:, :, None]

            # 计算 ROI 上边界：优先用 pose-aware，失败则回退到固定裁剪
            clothes_y1 = compute_pose_aware_roi_top(
                img, pose_model, pose_margin=pose_margin,
                bbox=(x1, y1, x2, y2)
            )

            if clothes_y1 is None:
                clothes_y1 = int(y1 + roi_top_ratio * (y2 - y1))

            clothes_y1 = max(y1, min(clothes_y1, y2 - 1))
            clothes_roi = person_region[clothes_y1:y2, x1:x2]

            if clothes_roi.size == 0:
                skipped_empty_roi += 1
                continue

            if clothes_roi.shape[0] < min_size or clothes_roi.shape[1] < min_size:
                skipped_small_roi += 1
                continue

            # 保存时移除实例索引，使用原始文件名
            save_name = f"{os.path.splitext(img_name)[0]}.jpg"
            save_path = os.path.join(output_folder, save_name)

            ok = cv2.imwrite(save_path, clothes_roi)
            if ok:
                saved_rois += 1
                if save_visualization:
                    cv2.rectangle(vis_img, (x1, clothes_y1), (x2, y2), (255, 140, 0), 2)

            if save_visualization and vis_img is not None:
                vis_name = f"{os.path.splitext(img_name)[0]}_seg.jpg"
                cv2.imwrite(os.path.join(visualization_dir, vis_name), vis_img)

        if mask_found_in_image:
            images_with_masks += 1

    return {
        "total_images": total_images,
        "unreadable_images": unreadable_images,
        "images_with_masks": images_with_masks,
        "total_instances": total_instances,
        "saved_rois": saved_rois,
        "skipped_small_roi": skipped_small_roi,
        "skipped_empty_roi": skipped_empty_roi,
        "segmenter_failures": segmenter_failures,
        "segmenter": segmenter.name,
        "roi_top_ratio": roi_top_ratio,
        "min_size": min_size,
    }


def save_stats(stats, output_folder):
    # 保存分割阶段的统计信息，后面写实验记录会用到。
    stats_path = os.path.join(output_folder, "segmentation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats_path


def build_parser():
    parser = argparse.ArgumentParser(description="Extract clothes ROI from person crops with segmentation model")
    parser.add_argument("--input_folder", type=str, default="dataset/person_crops", help="Input person crops folder")
    parser.add_argument("--output_folder", type=str, default="dataset/clothes_roi", help="Output clothes ROI folder")
    parser.add_argument(
        "--segmenter",
        type=str,
        default="yolov8",
        choices=["yolov8", "mask2former"],
        help="Segmentation model: yolov8 (YOLOv8-seg) or mask2former (Mask2Former)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path (for yolov8: local path, default yolov8x-seg.pt) or model name (for mask2former)",
    )
    parser.add_argument(
        "--roi_top_ratio",
        type=float,
        default=0.2,
        help="Top cutoff ratio inside detected person box (0~1)",
    )
    parser.add_argument("--min_size", type=int, default=20, help="Minimum ROI width/height")
    parser.add_argument("--save_vis", action="store_true", help="Save segmentation visualization")
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Visualization output directory (default: output_folder/vis)",
    )
    parser.add_argument("--pose_margin", type=int, default=50, help="Margin below shoulder keypoint (pixels) for ROI upper boundary")
    return parser


if __name__ == "__main__":
    # 命令行入口：便于直接切换输入目录和 ROI 裁切参数。
    parser = build_parser()
    args = parser.parse_args()

    # 根据参数segmenter来设置output_folder的默认值，保持结果有条理。
    args.output_folder = f"{args.output_folder}_{args.segmenter}"

    print("[分割] 参数已读取，开始运行服饰 ROI 提取脚本")

    stats = extract_clothes_roi(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        segmenter_name=args.segmenter,
        model_path=args.model_path,
        roi_top_ratio=args.roi_top_ratio,
        min_size=args.min_size,
        save_visualization=args.save_vis,
        visualization_dir=args.vis_dir,
        pose_margin=args.pose_margin,
    )
    stats_path = save_stats(stats, args.output_folder)

    print("Clothes ROIs extracted and saved to", args.output_folder)
    print("Stats saved to", stats_path)
    print("[分割] 运行完成")
    print(
        "Stats:",
        f"total_images={stats['total_images']},",
        f"unreadable_images={stats['unreadable_images']},",
        f"images_with_masks={stats['images_with_masks']},",
        f"total_instances={stats['total_instances']},",
        f"saved_rois={stats['saved_rois']},",
        f"skipped_small_roi={stats['skipped_small_roi']},",
        f"skipped_empty_roi={stats['skipped_empty_roi']},",
        f"segmenter_failures={stats['segmenter_failures']},",
        f"segmenter={stats['segmenter']}",
    )
