from ultralytics import YOLO
import cv2

import serial
import struct

from utils.camera_converter import CameraConverter  # 你之前写的相机坐标转换工具类


converter = CameraConverter()

depth_z = 1000  # 深度1000mm

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 摄像头/视频源
cap = cv2.VideoCapture(0)

# 串口（RS485）配置：按需修改
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

# 帧头帧尾（可按你的协议改）
FRAME_HEAD = b"\xAA\x55"
FRAME_TAIL = b"\x55\xAA"

# 发送类别：id=1
SEND_CLASS_ID = 1

# 打开串口
ser = None
if serial is not None:
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0,
        )
        print(f"串口已打开: {SERIAL_PORT} @ {BAUDRATE}")
    except Exception as e:
        print(f"串口打开失败: {e}")
        ser = None
else:
    print("未安装 pyserial，无法串口发送：请先运行 pip install pyserial")


def pick_leftmost_center_for_class(results0, class_id: int):
    """
    从单帧检测结果中，筛选指定class_id里“最左侧(x1最小)”目标，返回其中心点(cx, cy)。
    没有则返回 None。
    """
    boxes = getattr(results0, "boxes", None)
    if boxes is None or len(boxes) == 0 or boxes.cls is None:
        return None

    try:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (n,4)
    except Exception:
        return None

    best = None  # (x1, cx, cy)
    for i, c in enumerate(cls_ids):
        if c != class_id:
            continue
        x1, y1, x2, y2 = xyxy[i]
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        if best is None or x1 < best[0]:
            best = (x1, cx, cy)

    if best is None:
        return None

    _, cx, cy = best
    return cx, cy


def send_frame_center(cx: int, cy: int, w: int, h: int):
    """
    发送：HEAD(2) + X(float32 LE) + Y(float32 LE) + TAIL(2)
    """
    if ser is None:
        return

    # clamp到图像范围（避免越界）
    cx = max(0, min(w - 1, int(cx)))
    cy = max(0, min(h - 1, int(cy)))

    X, Y, Z = converter.pixel_to_world_coords(cx, cy, depth_z)

    payload = struct.pack("<ff", float(X), float(Y))
    frame = FRAME_HEAD + payload + FRAME_TAIL
    try:
        ser.write(frame)
    except Exception:
        pass


print("视频检测中，按'q'键退出...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 执行预测
    results = model(frame, conf=0.25)

    # 提取：id=1 最左侧目标中心点，并通过485发送（带帧头帧尾）
    picked = pick_leftmost_center_for_class(results[0], SEND_CLASS_ID)
    if picked is not None:
        cx, cy = picked
        send_frame_center(cx, cy, frame.shape[1], frame.shape[0])

    # 绘制检测结果
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if ser is not None:
    try:
        ser.close()
    except Exception:
        pass
