from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import argparse
import json


def iter_image_paths(input_folder, recursive=False):
    # 只遍历常见图片格式，避免把无关文件送进模型。
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if recursive:
        for root, _, files in os.walk(input_folder):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    yield os.path.join(root, name)
    else:
        for name in os.listdir(input_folder):
            path = os.path.join(input_folder, name)
            if os.path.isfile(path) and Path(name).suffix.lower() in exts:
                yield path


def draw_boxes(image, boxes, output_path):
    # 把检测结果画回原图，方便人工检查检测质量。
    vis = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            vis,
            f"person {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(output_path, vis)


def extract_person_crops(
    input_folder,
    output_folder,
    model_path="yolov8x.pt",
    conf=0.15,
    imgsz=1280,
    augment=True,
    recursive=False,
    save_visualization=False,
    visualization_dir=None,
):
    # 先准备输出目录，后面检测结果和统计信息都保存到这里。
    os.makedirs(output_folder, exist_ok=True)
    if save_visualization:
        if visualization_dir is None:
            visualization_dir = os.path.join(output_folder, "vis")
        os.makedirs(visualization_dir, exist_ok=True)

    model = YOLO(model_path)

    print(f"[检测] 开始处理：input={input_folder}, output={output_folder}, conf={conf}, imgsz={imgsz}, augment={augment}")

    total_images = 0
    unreadable_images = 0
    images_with_person = 0
    total_person_boxes = 0
    saved_crops = 0
    empty_detections = 0
    progress_step = 10

    for img_path in iter_image_paths(input_folder, recursive=recursive):
        total_images += 1
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            unreadable_images += 1
            continue

        if total_images == 1 or total_images % progress_step == 0:
            print(f"[检测] 已处理 {total_images} 张图片，当前累计裁切 {saved_crops} 张，检测到人物 {total_person_boxes} 个")

        results = model(img, conf=conf, imgsz=imgsz, augment=augment, verbose=False)
        person_in_this_image = 0
        person_boxes = []

        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 0 == person
                person_boxes.append(box)
                # 只保留人物框，并把框内区域裁切出来作为下一步输入。
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                #stem = Path(img_name).stem
                ok = cv2.imwrite(os.path.join(output_folder, f"{saved_crops+1}.jpg"), crop)
                if ok:
                    saved_crops += 1
                    person_in_this_image += 1

        if save_visualization:
            stem = Path(img_name).stem
            vis_path = os.path.join(visualization_dir, f"{stem}_det.jpg")
            draw_boxes(img, person_boxes, vis_path)

        if person_in_this_image > 0:
            images_with_person += 1
            total_person_boxes += person_in_this_image
        else:
            empty_detections += 1

    return {
        "total_images": total_images,                  # 处理了多少张图片
        "unreadable_images": unreadable_images,        # 无法读取的图片数量
        "images_with_person": images_with_person,      # 检测到人物的图片数量
        "images_without_person": empty_detections,     # 未检测到人物的图片数量
        "total_person_boxes": total_person_boxes,      # 检测到的总人物框数量
        "saved_crops": saved_crops,                    # 成功保存的人物裁剪数量
        "conf": conf,                                  # 检测置信度阈值
        "imgsz": imgsz,                                # 推理图像大小
        "augment": augment,                            # 是否启用推理增强
    }


def save_stats(stats, output_folder):
    # 把本次运行的关键统计写成 JSON，便于论文整理和复现。
    stats_path = os.path.join(output_folder, "detection_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats_path


def build_parser():
    parser = argparse.ArgumentParser(description="Extract person crops from images with YOLOv8")
    parser.add_argument("--input_folder", type=str, default="dataset\images", help="Input images folder (relative path)")
    parser.add_argument("--output_folder", type=str, default="dataset\person_crops", help="Output crops folder")
    parser.add_argument("--model_path", type=str, default="yolov8x.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.12, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--augment", action="store_true", help="Enable inference augmentation")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images")
    parser.add_argument("--save_vis", action="store_true", help="Save detection visualization")
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Visualization output directory (default: output_folder/vis)",
    )
    return parser


if __name__ == "__main__":
    # 命令行入口：可直接改输入目录、阈值和是否保存可视化。
    parser = build_parser()
    args = parser.parse_args()

    print("[检测] 参数已读取，开始运行人物检测脚本")

    stats = extract_person_crops(
        args.input_folder,
        args.output_folder,
        model_path=args.model_path,
        conf=args.conf,
        imgsz=args.imgsz,
        augment=args.augment,
        recursive=args.recursive,
        save_visualization=args.save_vis,
        visualization_dir=args.vis_dir,
    )
    stats_path = save_stats(stats, args.output_folder)
    print("Person crops extracted and saved to", args.output_folder)
    print("Stats saved to", stats_path)
    print("[检测] 运行完成")
    print(
        "Stats:",
        f"total_images={stats['total_images']},",
        f"unreadable_images={stats['unreadable_images']},",
        f"images_with_person={stats['images_with_person']},",
        f"images_without_person={stats['images_without_person']},",
        f"total_person_boxes={stats['total_person_boxes']},",
        f"saved_crops={stats['saved_crops']}",
    )