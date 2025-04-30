from ultralytics import YOLO
import netron

# 加载模型
model = YOLO("/home/zhihong/yolov11n/outputs/train1/road_obstacle_yolov11/weights/best.pt")  # 加载模型
 
# 导出模型
model.eval()
model.export(format='onnx', imgsz=(384, 640)) 
netron.start("/home/zhihong/yolov11n/outputs/train1/road_obstacle_yolov11/weights/best.onnx") #展示结构图