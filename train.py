from ultralytics import YOLO
import cv2

# 1. 加载预训练模型（推荐用yolov8n.pt，轻量化，训练快）
model = YOLO('yolov8n.pt')

# 2. 执行训练（参数根据你的数据集和设备调整）
results = model.train(
    data='vegetable.yaml',  # 自定义数据集配置文件
    epochs=100,               # 训练轮数，自定义数据集建议100-200
    batch=16,                 # 批量大小，根据GPU显存调整
    imgsz=640,                # 输入图像尺寸
    lr0=0.01,                 # 初始学习率
    device=0,                 # 0=GPU，-1=CPU
    patience=100,             # 早停耐心值
    save=True,                # 保存最佳模型
    project='vegetable_train',   # 项目保存路径
    name='yolov8n_vegetable',    # 实验名称
    pretrained=True,          # 迁移学习，加速收敛
    cache=True,               # 缓存数据
    augment=True,             # 开启数据增强（提升模型泛化能力，避免过拟合）
    hsv_h=0.015,              # 色调增强（数据增强参数，默认即可）
    hsv_s=0.7,                # 饱和度增强
    hsv_v=0.4,                # 明度增强
    degrees=0.0,              # 旋转角度（0表示不旋转，可改为10增强）
    flipud=0.0,               # 上下翻转概率
    fliplr=0.5,               # 左右翻转概率（0.5表示50%概率翻转）
    mosaic=1.0                # Mosaic数据增强（默认开启，提升模型鲁棒性）
)

# 3. 模型评估
val_results = model.val()
print(f"自定义数据集训练完成！mAP@0.5：{val_results.box.map:.4f}")

# 4. 模型预测
model = YOLO('custom_train/yolov8n_custom/weights/best.pt')  # 加载最佳模型
test_img_path = 'custom_test.jpg'  # 自定义测试图像路径
results = model(test_img_path, conf=0.25)
annotated_img = results[0].plot()
cv2.imshow('Custom Dataset Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('custom_result.jpg', annotated_img)
print("自定义数据集预测结果已保存为custom_result.jpg")
