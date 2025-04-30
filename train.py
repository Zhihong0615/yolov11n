from ultralytics import YOLO

def main():
    
    model = YOLO("/home/zhihong/yolov11n/models/yolo11.yaml")
    model.load("/home/zhihong/yolov11n/models/yolo11n.pt") # 加载预训练模型权重

    # 开始训练
    model.train(
        data="/home/zhihong/yolov11n/dataset/data.yaml",  # 配置文件路径
        epochs=100,                                  # 训练轮数
        imgsz=640,                                  # 输入图像大小
        batch=16,                                   # 每批图像数量（视GPU调整）
        name="road_obstacle_yolov11",               # 本次实验名称
        project="/home/zhihong/yolov11n/outputs/train1",                 # 输出目录
        resume=False,                               # 重新训练设置为True
        device=0                                    # 使用 GPU，写 -1 为 CPU
    )

if __name__ == '__main__':
    main()
