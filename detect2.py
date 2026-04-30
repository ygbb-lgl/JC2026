import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ==========================
# 参数配置
# ==========================
MODEL_PATH = "runs/train/spinach_yolo/weights/best.pt"
CAMERA_ID = 0
IMAGE_PATH = "test.jpg"
MODE = "camera"  # 可选："camera" 或 "image"
TARGET_CLASS_NAME = "bocai"  # YOLO 模型中的菠菜类别名称；留空则接受所有类别

# bbox 扩展比例
BBOX_EXPAND_RATIO = 0.05
BBOX_BOTTOM_EXTRAPOLATION = 0.2

# 颜色空间选择：'HSV'、'LAB' 或 'BOTH'
PLANT_MASK_SPACE = "BOTH"
LEAF_MASK_SPACE = "BOTH"

# 植物整体 mask 阈值（HSV）
PLANT_HSV_LOWER = [0, 20, 10]
PLANT_HSV_UPPER = [180, 255, 255]

# 植物整体 mask 阈值（LAB）
PLANT_LAB_LOWER = [0, 110, 110]
PLANT_LAB_UPPER = [255, 170, 200]

# 绿色叶片阈值（HSV）
LEAF_HSV_LOWER = [30, 40, 40]
LEAF_HSV_UPPER = [90, 255, 255]

# 绿色叶片阈值（LAB）
LEAF_LAB_LOWER = [20, 130, 110]
LEAF_LAB_UPPER = [255, 175, 170]

# 形态学处理参数
STEM_CLOSE_KERNEL = 7
STEM_CLOSE_ITER = 2
STEM_OPEN_KERNEL = 5
STEM_OPEN_ITER = 1
STEM_DILATE_ITER = 2
STEM_ERODE_ITER = 1

# 根茎筛选参数
MIN_STEM_AREA = 200
MAX_STEM_AREA_RATIO = 0.22
MIN_ASPECT_RATIO = 1.8
MAX_EXTENT = 0.75
MIN_SOLIDITY = 0.30
MIN_BOTTOM_FRACTION = 0.40
BOTTOM_REGION_RATIO = 0.30
# 输出路径
OUTPUT_ROOT = Path("output")
DEBUG_OUTPUT = OUTPUT_ROOT / "debug"

# 显示窗口名称
WINDOW_NAMES = {
    "original": "Original",
    "yolo": "YOLO Detection",
    "roi": "ROI",
    "plant_mask": "Plant Mask",
    "leaf_mask": "Leaf Mask",
    "stem_candidate": "Stem Candidate",
    "stem_mask": "Final Stem Mask",
    "overlay": "Stem Overlay",
}


# ==========================
# 工具函数
# ==========================

def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    model = YOLO(model_path)
    return model


def expand_bbox(x1: int, y1: int, x2: int, y2: int, image_shape, expand_ratio: float, bottom_ratio: float):
    h, w = image_shape[:2]
    width = x2 - x1
    height = y2 - y1
    dx = int(width * expand_ratio)
    dy = int(height * expand_ratio)
    new_x1 = max(0, x1 - dx)
    new_x2 = min(w, x2 + dx)
    new_y1 = max(0, y1 - dy)
    new_y2 = min(h, int(y2 + dy + height * bottom_ratio))
    return new_x1, new_y1, new_x2, new_y2


def crop_roi(image, bbox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def get_plant_mask(roi, params):
    masks = []
    if params["plant_mask_space"] in ("HSV", "BOTH"):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(params["plant_hsv_lower"], dtype=np.uint8)
        upper = np.array(params["plant_hsv_upper"], dtype=np.uint8)
        masks.append(cv2.inRange(hsv, lower, upper))
    if params["plant_mask_space"] in ("LAB", "BOTH"):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        lower = np.array(params["plant_lab_lower"], dtype=np.uint8)
        upper = np.array(params["plant_lab_upper"], dtype=np.uint8)
        masks.append(cv2.inRange(lab, lower, upper))
    if not masks:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    mask = masks[0]
    for extra in masks[1:]:
        mask = cv2.bitwise_or(mask, extra)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def get_leaf_mask_hsv(roi, params):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array(params["leaf_hsv_lower"], dtype=np.uint8)
    upper = np.array(params["leaf_hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def get_leaf_mask_lab(roi, params):
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    lower = np.array(params["leaf_lab_lower"], dtype=np.uint8)
    upper = np.array(params["leaf_lab_upper"], dtype=np.uint8)
    mask = cv2.inRange(lab, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def get_leaf_mask(roi, params):
    masks = []
    if params["leaf_mask_space"] in ("HSV", "BOTH"):
        masks.append(get_leaf_mask_hsv(roi, params))
    if params["leaf_mask_space"] in ("LAB", "BOTH"):
        masks.append(get_leaf_mask_lab(roi, params))
    if not masks:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    mask = masks[0]
    for extra in masks[1:]:
        mask = cv2.bitwise_or(mask, extra)
    return mask


def extract_stem_candidate(plant_mask, leaf_mask, params):
    leaf_inv = cv2.bitwise_not(leaf_mask)
    stem_candidate = cv2.bitwise_and(plant_mask, leaf_inv)
    if stem_candidate.max() == 0:
        return stem_candidate
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params["stem_close_kernel"], params["stem_close_kernel"]))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params["stem_open_kernel"], params["stem_open_kernel"]))
    stem_candidate = cv2.morphologyEx(stem_candidate, cv2.MORPH_CLOSE, close_kernel, iterations=params["stem_close_iter"])
    stem_candidate = cv2.morphologyEx(stem_candidate, cv2.MORPH_OPEN, open_kernel, iterations=params["stem_open_iter"])
    if params["stem_dilate_iter"] > 0:
        stem_candidate = cv2.dilate(stem_candidate, open_kernel, iterations=params["stem_dilate_iter"])
    if params["stem_erode_iter"] > 0:
        stem_candidate = cv2.erode(stem_candidate, open_kernel, iterations=params["stem_erode_iter"])
    # 去除小噪声并保持候选区域连通性
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stem_candidate = cv2.morphologyEx(stem_candidate, cv2.MORPH_OPEN, small_kernel, iterations=1)
    return stem_candidate


def filter_stem_components(stem_candidate, roi_shape, params):
    stem_mask = np.zeros_like(stem_candidate)
    contours, _ = cv2.findContours(stem_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return stem_mask
    roi_area = roi_shape[0] * roi_shape[1]
    selected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params["min_stem_area"]:
            continue
        if area > params["max_stem_area_ratio"] * roi_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if aspect_ratio < params["min_aspect_ratio"]:
            continue
        rect_area = w * h
        extent = float(area) / (rect_area + 1e-5)
        if extent > params["max_extent"]:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / (hull_area + 1e-5)
        if solidity < params["min_solidity"]:
            continue
        bottom_fraction = float(y + h) / roi_shape[0]
        bottom_overlap = (y + h) >= roi_shape[0] * (1.0 - params["bottom_region_ratio"])
        if bottom_fraction < params["min_bottom_fraction"] and not bottom_overlap:
            continue
        score = area * (0.5 + bottom_fraction)
        if bottom_overlap:
            score *= 1.3
        selected.append((score, cnt))
    selected.sort(key=lambda item: item[0], reverse=True)
    for _, cnt in selected:
        cv2.drawContours(stem_mask, [cnt], -1, 255, thickness=-1)
    return stem_mask


def draw_bboxes(frame, bboxes):
    output = frame.copy()
    for idx, (x1, y1, x2, y2, conf, label) in enumerate(bboxes, start=1):
        color = (255, 180, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(output, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output


def build_roi_preview(image, bboxes):
    rois = []
    for bbox in bboxes:
        x1, y1, x2, y2, *_ = bbox
        roi = crop_roi(image, (x1, y1, x2, y2))
        if roi is not None:
            rois.append(roi)
    if not rois:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    max_h = max(roi.shape[0] for roi in rois)
    preview = np.zeros((max_h, sum(roi.shape[1] for roi in rois), 3), dtype=np.uint8)
    current_x = 0
    for roi in rois:
        h, w = roi.shape[:2]
        preview[0:h, current_x:current_x + w] = roi
        current_x += w
    return preview


def create_overlay(frame, stem_mask_full):
    overlay = frame.copy()
    color_layer = np.zeros_like(frame)
    color_layer[stem_mask_full > 0] = (0, 255, 255)
    overlay = cv2.addWeighted(frame, 0.7, color_layer, 0.3, 0)
    return overlay


def save_debug_images(images, prefix):
    DEBUG_OUTPUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for key, image in images.items():
        filename = DEBUG_OUTPUT / f"{prefix}_{timestamp}_{key}.png"
        cv2.imwrite(str(filename), image)
    print(f"已保存调试图像到 {DEBUG_OUTPUT}")


def generate_full_mask_display(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def process_frame(frame, model, params):
    result = {
        "original": frame.copy(),
        "yolo": frame.copy(),
        "roi": np.zeros((200, 400, 3), dtype=np.uint8),
        "plant_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "leaf_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "stem_candidate": np.zeros(frame.shape[:2], dtype=np.uint8),
        "stem_mask": np.zeros(frame.shape[:2], dtype=np.uint8),
        "overlay": frame.copy(),
        "has_detection": False,
    }

    try:
        predictions = model(frame)
    except Exception as exc:
        print("YOLO 推理失败：", exc)
        return result

    if len(predictions) == 0 or predictions[0].boxes.shape[0] == 0:
        return result

    box_data = predictions[0].boxes.data
    if box_data is None or len(box_data) == 0:
        return result
    box_data = box_data.cpu().numpy()

    bboxes = []
    for det in box_data:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = str(int(cls_id))
        if hasattr(model, "names") and int(cls_id) in model.names:
            label = model.names[int(cls_id)]
        if TARGET_CLASS_NAME and TARGET_CLASS_NAME.lower() not in label.lower():
            continue
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, frame.shape, params["bbox_expand_ratio"], params["bbox_bottom_extrapolation"])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes.append((x1, y1, x2, y2, conf, label))

    if not bboxes:
        return result

    result["has_detection"] = True
    result["yolo"] = draw_bboxes(result["yolo"], bboxes)
    result["roi"] = build_roi_preview(frame, bboxes)

    for bbox_idx, bbox in enumerate(bboxes, start=1):
        x1, y1, x2, y2, _, _ = bbox
        roi = crop_roi(frame, (x1, y1, x2, y2))
        if roi is None:
            continue
        plant_mask = get_plant_mask(roi, params)
        leaf_mask = get_leaf_mask(roi, params)
        stem_candidate = extract_stem_candidate(plant_mask, leaf_mask, params)
        stem_mask = filter_stem_components(stem_candidate, roi.shape, params)

        result["plant_mask"][y1:y2, x1:x2] = cv2.bitwise_or(result["plant_mask"][y1:y2, x1:x2], plant_mask)
        result["leaf_mask"][y1:y2, x1:x2] = cv2.bitwise_or(result["leaf_mask"][y1:y2, x1:x2], leaf_mask)
        result["stem_candidate"][y1:y2, x1:x2] = cv2.bitwise_or(result["stem_candidate"][y1:y2, x1:x2], stem_candidate)
        result["stem_mask"][y1:y2, x1:x2] = cv2.bitwise_or(result["stem_mask"][y1:y2, x1:x2], stem_mask)

    result["overlay"] = create_overlay(frame, result["stem_mask"])
    return result


def show_windows(debug_images):
    for key, image in debug_images.items():
        cv2.imshow(WINDOW_NAMES.get(key, key), image)


def save_results(debug_images, prefix):
    save_debug_images(debug_images, prefix)


def load_parameters():
    return {
        "bbox_expand_ratio": BBOX_EXPAND_RATIO,
        "bbox_bottom_extrapolation": BBOX_BOTTOM_EXTRAPOLATION,
        "plant_mask_space": PLANT_MASK_SPACE,
        "leaf_mask_space": LEAF_MASK_SPACE,
        "plant_hsv_lower": PLANT_HSV_LOWER,
        "plant_hsv_upper": PLANT_HSV_UPPER,
        "plant_lab_lower": PLANT_LAB_LOWER,
        "plant_lab_upper": PLANT_LAB_UPPER,
        "leaf_hsv_lower": LEAF_HSV_LOWER,
        "leaf_hsv_upper": LEAF_HSV_UPPER,
        "leaf_lab_lower": LEAF_LAB_LOWER,
        "leaf_lab_upper": LEAF_LAB_UPPER,
        "stem_close_kernel": STEM_CLOSE_KERNEL,
        "stem_close_iter": STEM_CLOSE_ITER,
        "stem_open_kernel": STEM_OPEN_KERNEL,
        "stem_open_iter": STEM_OPEN_ITER,
        "stem_dilate_iter": STEM_DILATE_ITER,
        "stem_erode_iter": STEM_ERODE_ITER,
        "min_stem_area": MIN_STEM_AREA,
        "max_stem_area_ratio": MAX_STEM_AREA_RATIO,
        "min_aspect_ratio": MIN_ASPECT_RATIO,
        "max_extent": MAX_EXTENT,
        "min_solidity": MIN_SOLIDITY,
        "min_bottom_fraction": MIN_BOTTOM_FRACTION,
        "bottom_region_ratio": BOTTOM_REGION_RATIO,
    }


def main():
    params = load_parameters()
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        print(f"加载模型失败: {exc}")
        return

    if MODE == "camera":
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print(f"无法打开摄像头: {CAMERA_ID}")
            return
        paused = False
        frame_idx = 0
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("摄像头读取失败或已结束")
                    break
                frame_idx += 1
                debug_images = process_frame(frame, model, params)
                show_windows({
                    "original": debug_images["original"],
                    "yolo": debug_images["yolo"],
                    "roi": debug_images["roi"],
                    "plant_mask": generate_full_mask_display(debug_images["plant_mask"]),
                    "leaf_mask": generate_full_mask_display(debug_images["leaf_mask"]),
                    "stem_candidate": generate_full_mask_display(debug_images["stem_candidate"]),
                    "stem_mask": generate_full_mask_display(debug_images["stem_mask"]),
                    "overlay": debug_images["overlay"],
                })
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and not paused:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_results({
                    "original": debug_images["original"],
                    "yolo": debug_images["yolo"],
                    "roi": debug_images["roi"],
                    "plant_mask": generate_full_mask_display(debug_images["plant_mask"]),
                    "leaf_mask": generate_full_mask_display(debug_images["leaf_mask"]),
                    "stem_candidate": generate_full_mask_display(debug_images["stem_candidate"]),
                    "stem_mask": generate_full_mask_display(debug_images["stem_mask"]),
                    "overlay": debug_images["overlay"],
                }, f"camera_{timestamp}")
            if key in (ord(" "), 13):
                paused = not paused
        cap.release()
    elif MODE == "image":
        if not os.path.exists(IMAGE_PATH):
            print(f"图像路径不存在: {IMAGE_PATH}")
            return
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            print(f"无法读取图像: {IMAGE_PATH}")
            return
        debug_images = process_frame(image, model, params)
        show_windows({
            "original": debug_images["original"],
            "yolo": debug_images["yolo"],
            "roi": debug_images["roi"],
            "plant_mask": generate_full_mask_display(debug_images["plant_mask"]),
            "leaf_mask": generate_full_mask_display(debug_images["leaf_mask"]),
            "stem_candidate": generate_full_mask_display(debug_images["stem_candidate"]),
            "stem_mask": generate_full_mask_display(debug_images["stem_mask"]),
            "overlay": debug_images["overlay"],
        })
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_results({
                    "original": debug_images["original"],
                    "yolo": debug_images["yolo"],
                    "roi": debug_images["roi"],
                    "plant_mask": generate_full_mask_display(debug_images["plant_mask"]),
                    "leaf_mask": generate_full_mask_display(debug_images["leaf_mask"]),
                    "stem_candidate": generate_full_mask_display(debug_images["stem_candidate"]),
                    "stem_mask": generate_full_mask_display(debug_images["stem_mask"]),
                    "overlay": debug_images["overlay"],
                }, "image")
            if key in (ord(" "), 13):
                break
    else:
        print(f"未知运行模式: {MODE}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
