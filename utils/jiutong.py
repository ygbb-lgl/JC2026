import cv2
import numpy as np
import time


def nothing(x):
    pass


def ensure_odd(x, minimum=1):
    if x < minimum:
        x = minimum
    if x % 2 == 0:
        x += 1
    return x


def remove_small_components(mask, min_area=80):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_complexity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if area < 1:
        return 0.0
    return (peri * peri) / area


def pca_axis_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    center = pts.mean(axis=0)

    pts_centered = pts - center
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    main_axis = eigvecs[:, 0]

    norm = np.linalg.norm(main_axis)
    if norm < 1e-6:
        return None

    main_axis = main_axis / norm
    return center, main_axis


def project_points(mask, center, axis):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, None

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    rel = pts - center
    proj = rel @ axis
    return xs, ys, proj


def make_end_mask(mask, center, axis, low_ratio, high_ratio):
    xs, ys, proj = project_points(mask, center, axis)
    out = np.zeros_like(mask)

    if proj is None or len(proj) == 0:
        return out

    pmin = proj.min()
    pmax = proj.max()
    if abs(pmax - pmin) < 1e-6:
        return out

    low = pmin + (pmax - pmin) * low_ratio
    high = pmin + (pmax - pmin) * high_ratio
    keep = (proj >= low) & (proj <= high)

    out[ys[keep], xs[keep]] = 255
    return out


def axis_length(mask, center, axis):
    xs, ys, proj = project_points(mask, center, axis)
    if proj is None or len(proj) == 0:
        return 1.0
    return float(proj.max() - proj.min() + 1e-6)


def width_proxy(mask, center, axis):
    area = cv2.countNonZero(mask)
    length = axis_length(mask, center, axis)
    if length < 1e-6:
        return 0.0
    return area / length


def classify_component(comp_mask, aspect_thr, tip_cmp_thr, tip_width_ratio_thr):
    cnt = get_largest_contour(comp_mask)
    if cnt is None:
        return "UNKNOWN", {}, np.zeros_like(comp_mask)

    area = cv2.contourArea(cnt)
    if area < 20:
        return "UNKNOWN", {}, np.zeros_like(comp_mask)

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), _ = rect
    long_side = max(rw, rh)
    short_side = max(1.0, min(rw, rh))
    aspect_ratio = long_side / short_side

    pca_res = pca_axis_from_mask(comp_mask)
    if pca_res is None:
        return "UNKNOWN", {
            "aspect_ratio": aspect_ratio,
            "tip_complexity": 0.0,
            "tip_width_ratio": 0.0
        }, np.zeros_like(comp_mask)

    center, axis = pca_res

    end_a = make_end_mask(comp_mask, center, axis, 0.00, 0.20)
    end_b = make_end_mask(comp_mask, center, axis, 0.80, 1.00)

    cnt_a = get_largest_contour(end_a)
    cnt_b = get_largest_contour(end_b)

    cmp_a = contour_complexity(cnt_a) if cnt_a is not None else 0.0
    cmp_b = contour_complexity(cnt_b) if cnt_b is not None else 0.0

    if cmp_b > cmp_a:
        tip_mask = end_b
        tip_complexity = cmp_b
    else:
        tip_mask = end_a
        tip_complexity = cmp_a

    whole_width = width_proxy(comp_mask, center, axis)
    tip_width = width_proxy(tip_mask, center, axis)
    tip_width_ratio = tip_width / max(1e-6, whole_width)

    score_jiu = 0
    score_tong = 0

    if aspect_ratio >= aspect_thr:
        score_jiu += 2
    elif aspect_ratio >= aspect_thr * 0.9:
        score_jiu += 1

    if tip_complexity >= tip_cmp_thr:
        score_tong += 2
    elif tip_complexity >= tip_cmp_thr * 0.85:
        score_tong += 1

    if tip_width_ratio >= tip_width_ratio_thr:
        score_tong += 2
    elif tip_width_ratio >= tip_width_ratio_thr * 0.9:
        score_tong += 1

    if tip_width_ratio < 1.6:
        score_jiu += 1

    if score_jiu >= score_tong + 1:
        label = "JIUCAI"
    elif score_tong >= score_jiu + 1:
        label = "TONGHAO"
    else:
        label = "UNKNOWN"

    feat = {
        "aspect_ratio": float(aspect_ratio),
        "tip_complexity": float(tip_complexity),
        "tip_width_ratio": float(tip_width_ratio),
        "score_jiu": int(score_jiu),
        "score_tong": int(score_tong),
    }

    return label, feat, tip_mask


def draw_label_box(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x1 = x
    y1 = max(0, y - th - 10)
    x2 = x + tw + 10
    y2 = y

    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, scale, (0, 0, 0), thickness)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("controls", 540, 640)

    # ========= 分割参数 =========
    cv2.createTrackbar("L_min", "controls", 135, 255, nothing)
    cv2.createTrackbar("L_max", "controls", 205, 255, nothing)
    cv2.createTrackbar("A_min", "controls", 123, 255, nothing)
    cv2.createTrackbar("A_max", "controls", 145, 255, nothing)
    cv2.createTrackbar("B_min", "controls", 72, 255, nothing)
    cv2.createTrackbar("B_max", "controls", 100, 255, nothing)

    cv2.createTrackbar("GaussianK", "controls", 5, 31, nothing)
    cv2.createTrackbar("OpenK", "controls", 3, 21, nothing)
    cv2.createTrackbar("CloseK", "controls", 5, 21, nothing)
    cv2.createTrackbar("MinArea", "controls", 80, 5000, nothing)

    # ========= 分类参数 =========
    cv2.createTrackbar("Aspect_Jiu_x10", "controls", 30, 150, nothing)       # 3.0
    cv2.createTrackbar("TipCmp_Tong", "controls", 40, 300, nothing)           # 40
    cv2.createTrackbar("TipWidthRatio_x10", "controls", 18, 100, nothing)     # 1.8
    cv2.createTrackbar("MinAreaCls", "controls", 800, 10000, nothing)

    # ========= 新增：连通域分离参数 =========
    cv2.createTrackbar("TopMargin", "controls", 40, 200, nothing)
    cv2.createTrackbar("SideMargin", "controls", 12, 100, nothing)
    cv2.createTrackbar("BottomMargin", "controls", 12, 100, nothing)
    cv2.createTrackbar("ClsErodeK", "controls", 3, 9, nothing)
    cv2.createTrackbar("ClsErodeIter", "controls", 1, 3, nothing)

    prev_t = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头失败")
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # ===== 读取分割参数 =====
        l_min = cv2.getTrackbarPos("L_min", "controls")
        l_max = cv2.getTrackbarPos("L_max", "controls")
        a_min = cv2.getTrackbarPos("A_min", "controls")
        a_max = cv2.getTrackbarPos("A_max", "controls")
        b_min = cv2.getTrackbarPos("B_min", "controls")
        b_max = cv2.getTrackbarPos("B_max", "controls")

        gk = ensure_odd(cv2.getTrackbarPos("GaussianK", "controls"), 1)
        open_k = ensure_odd(cv2.getTrackbarPos("OpenK", "controls"), 1)
        close_k = ensure_odd(cv2.getTrackbarPos("CloseK", "controls"), 1)
        min_area = max(1, cv2.getTrackbarPos("MinArea", "controls"))

        # ===== 读取分类参数 =====
        aspect_thr = cv2.getTrackbarPos("Aspect_Jiu_x10", "controls") / 10.0
        tip_cmp_thr = float(cv2.getTrackbarPos("TipCmp_Tong", "controls"))
        tip_width_ratio_thr = cv2.getTrackbarPos("TipWidthRatio_x10", "controls") / 10.0
        min_area_cls = max(1, cv2.getTrackbarPos("MinAreaCls", "controls"))

        # ===== 读取新增分离参数 =====
        top_margin = cv2.getTrackbarPos("TopMargin", "controls")
        side_margin = cv2.getTrackbarPos("SideMargin", "controls")
        bottom_margin = cv2.getTrackbarPos("BottomMargin", "controls")
        cls_erode_k = ensure_odd(cv2.getTrackbarPos("ClsErodeK", "controls"), 1)
        cls_erode_iter = max(0, cv2.getTrackbarPos("ClsErodeIter", "controls"))

        # ===== 1. 语义分割 =====
        blurred = cv2.GaussianBlur(frame, (gk, gk), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)

        lower_bg = np.array([l_min, a_min, b_min], dtype=np.uint8)
        upper_bg = np.array([l_max, a_max, b_max], dtype=np.uint8)

        bg_mask = cv2.inRange(lab, lower_bg, upper_bg)
        plant_mask = cv2.bitwise_not(bg_mask)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel_open)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_close)
        plant_mask = remove_small_components(plant_mask, min_area=min_area)

        # ===== 2. 边缘裁切（显示用 mask） =====
        plant_mask[:top_margin, :] = 0
        plant_mask[-bottom_margin:, :] = 0
        plant_mask[:, :side_margin] = 0
        plant_mask[:, -side_margin:] = 0

        # ===== 3. 分类专用 mask：轻微腐蚀，断开细连接 =====
        plant_mask_cls = plant_mask.copy()
        if cls_erode_iter > 0 and cls_erode_k >= 1:
            cls_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cls_erode_k, cls_erode_k))
            plant_mask_cls = cv2.erode(plant_mask_cls, cls_kernel, iterations=cls_erode_iter)

        # ===== 4. 连通域分类 =====
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(plant_mask_cls, connectivity=8)

        vis = frame.copy()
        mask_bgr = cv2.cvtColor(plant_mask_cls, cv2.COLOR_GRAY2BGR)

        feature_lines = []
        shown = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area_cls:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            img_area = H * W
            area_ratio = area / float(img_area)
            touches_border = (x <= 2) or (y <= 2) or (x + w >= W - 2) or (y + h >= H - 2)

            if area_ratio > 0.20:
                continue

            if touches_border and area_ratio > 0.01:
                continue

            comp_mask = np.zeros_like(plant_mask_cls)
            comp_mask[labels == i] = 255

            label, feat, tip_mask = classify_component(
                comp_mask,
                aspect_thr,
                tip_cmp_thr,
                tip_width_ratio_thr
            )

            if label == "JIUCAI":
                color = (0, 255, 0)
            elif label == "TONGHAO":
                color = (0, 165, 255)
            else:
                color = (160, 160, 160)

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            draw_label_box(vis, label, x, max(28, y), color)

            tip_cnt = get_largest_contour(tip_mask)
            if tip_cnt is not None:
                cv2.drawContours(vis, [tip_cnt], -1, (0, 255, 255), 2)

            if shown < 10:
                feature_lines.append(
                    f"{label} asp={feat.get('aspect_ratio',0):.2f} tipCmp={feat.get('tip_complexity',0):.1f} tipWR={feat.get('tip_width_ratio',0):.2f}"
                )
                shown += 1

        # ===== 5. FPS =====
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        # ===== 6. 信息栏 =====
        info_panel = np.zeros((250, W * 2, 3), dtype=np.uint8)

        info1 = f"BG LAB=[{l_min},{l_max}] [{a_min},{a_max}] [{b_min},{b_max}]"
        info2 = f"Aspect_Jiu={aspect_thr:.1f}  TipCmp_Tong={tip_cmp_thr:.1f}  TipWidthRatio={tip_width_ratio_thr:.1f}"
        info3 = f"MinAreaCls={min_area_cls}  FPS={fps:.1f}"
        info4 = f"TopMargin={top_margin}  SideMargin={side_margin}  BottomMargin={bottom_margin}"
        info5 = f"ClsErodeK={cls_erode_k}  ClsErodeIter={cls_erode_iter}"

        cv2.putText(info_panel, info1, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, info2, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, info3, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, info4, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(info_panel, info5, (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

        cv2.putText(info_panel, "JIUCAI=Green  TONGHAO=Orange  UNKNOWN=Gray  TIP=Yellow",
                    (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        y0 = 240
        for idx, line in enumerate(feature_lines[:8]):
            cv2.putText(info_panel, line, (20 + (idx // 4) * 760, y0 + (idx % 4) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

        # ===== 7. 拼图 =====
        top = np.hstack([vis, mask_bgr])
        final_show = np.vstack([top, info_panel])

        cv2.putText(final_show, "Left: Original + Classification", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(final_show, "Right: Semantic Mask after top-cut + erode", (W + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

        cv2.imshow("tongjiu2 morph v6 separated", final_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            cv2.imwrite("classified_view.png", vis)
            cv2.imwrite("semantic_mask_cls.png", plant_mask_cls)
            cv2.imwrite("info_panel.png", info_panel)
            print("已保存: classified_view.png / semantic_mask_cls.png / info_panel.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()