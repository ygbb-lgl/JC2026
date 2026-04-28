from ultralytics import YOLO
import cv2

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 选项1：检测本地视频（替换为你的视频路径，支持mp4、avi等格式）
cap = cv2.VideoCapture(0)

# 选项2：摄像头实时检测（将video_path改为0，电脑默认摄像头；1为外接摄像头）
# cap = cv2.VideoCapture(0)

# 获取视频的宽度、高度、帧率（用于保存结果视频）
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 创建视频写入器（保存检测结果）
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('video_result.mp4', fourcc, fps, (width, height))

print("视频检测中，按'q'键退出...")

# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕，退出循环

    # 执行预测
    results = model(frame, conf=0.25)
    # 绘制检测结果
    annotated_frame = results[0].plot()

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
