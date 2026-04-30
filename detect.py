from ultralytics import YOLO
import cv2
import time
import numpy as np

# 加载预训练模型
model = YOLO('best.pt')

# 选项1：检测本地视频（替换为你的视频路径，支持mp4、avi等格式）
# cap = cv2.VideoCapture('your_video.mp4')

# 选项2：摄像头实时检测（将video_path改为0，电脑默认摄像头；1为外接摄像头）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('无法打开摄像头，请检查设备或改成本地视频路径')
    exit(1)

# 获取视频的宽度、高度、帧率（用于保存结果视频）
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 创建视频写入器（保存检测结果）
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('video_result.mp4', fourcc, fps, (width, height))

print("视频检测中，按'q'键退出...")

prev_time = time.time()

# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕，退出循环

    # 执行预测
    results = model(frame, conf=0.25)
    # 绘制检测结果
    annotated_frame = results[0].plot()

    # 处理 YOLO 检测框中心点显示
    try:
        box_data = results[0].boxes.data
        if box_data is not None and len(box_data) > 0:
            for det in box_data:
                det_np = det.cpu().numpy() if hasattr(det, 'cpu') else det
                x1, y1, x2, y2, conf, cls_id = det_np[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(annotated_frame, f"box_center={cx},{cy}", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    except Exception as e:
        print('检测框中心点绘制错误:', e)

    # 计算实时帧率（FPS）
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示实时检测画面
    cv2.imshow('YOLOv8 Video Detection', annotated_frame)

    # 保存检测帧到结果视频
    # out.write(annotated_frame)

    # 按'q'键退出（等待1ms，控制视频播放速度）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
# out.release()
cv2.destroyAllWindows()
print("视频检测完成，结果已保存为video_result.mp4")
