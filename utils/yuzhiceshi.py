import cv2
import numpy as np
import argparse


def on_mouse(event, x, y, flags, param):
    data = param
    img = data["img"]
    img_show = data["img_show"]
    lab = data["lab"]
    hsv = data["hsv"]

    if event == cv2.EVENT_LBUTTONDOWN:
        # OpenCV 默认是 BGR
        b, g, r = img[y, x]
        l, a, bb = lab[y, x]
        h, s, v = hsv[y, x]

        print("-" * 70)
        print(f"坐标 (x, y) = ({x}, {y})")
        print(f"BGR = ({int(b)}, {int(g)}, {int(r)})")
        print(f"RGB = ({int(r)}, {int(g)}, {int(b)})")
        print(f"LAB = ({int(l)}, {int(a)}, {int(bb)})")
        print(f"HSV = ({int(h)}, {int(s)}, {int(v)})")

        # 刷新显示图
        data["img_show"] = img.copy()
        img_show = data["img_show"]

        # 画十字和圆点
        cv2.circle(img_show, (x, y), 5, (0, 0, 255), 2)
        cv2.line(img_show, (x - 12, y), (x + 12, y), (0, 255, 255), 1)
        cv2.line(img_show, (x, y - 12), (x, y + 12), (0, 255, 255), 1)

        # 显示信息文字
        text1 = f"XY=({x},{y})"
        text2 = f"RGB=({int(r)},{int(g)},{int(b)})"
        text3 = f"LAB=({int(l)},{int(a)},{int(bb)})"
        text4 = f"HSV=({int(h)},{int(s)},{int(v)})"

        tx = x + 15
        ty = y - 15

        # 防止文字超出边界
        h_img, w_img = img.shape[:2]
        if tx > w_img - 260:
            tx = x - 260
        if ty < 60:
            ty = y + 30

        cv2.rectangle(img_show, (tx - 5, ty - 20), (tx + 250, ty + 65), (0, 0, 0), -1)
        cv2.putText(img_show, text1, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img_show, text2, (tx, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img_show, text3, (tx, ty + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(img_show, text4, (tx, ty + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="点击图片查看某个像素点的 RGB/LAB/HSV 值")
    parser.add_argument("--image", required=True, help="输入图片路径，例如 ./test.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"图片读取失败: {args.image}")
        return

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    data = {
        "img": img,
        "img_show": img.copy(),
        "lab": lab,
        "hsv": hsv,
    }

    cv2.namedWindow("pixel_probe", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pixel_probe", on_mouse, data)

    print("使用说明：")
    print("1. 鼠标左键点击任意像素点")
    print("2. 终端会打印该点的 BGR / RGB / LAB / HSV")
    print("3. 按 q 或 ESC 退出")
    print("-" * 70)

    while True:
        cv2.imshow("pixel_probe", data["img_show"])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            data["img_show"] = data["img"].copy()
            print("显示已重置")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()