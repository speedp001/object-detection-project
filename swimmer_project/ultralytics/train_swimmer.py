from ultralytics import YOLO

if __name__ == "__main__" :
    
    model = YOLO("yolov8m.pt")
    
    model.train(data="swimmer.yaml", epochs=100, batch=8, hsv_s=0, hsv_v=0, degrees=5, lrf=0.0025, device="mps")