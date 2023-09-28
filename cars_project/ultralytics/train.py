from ultralytics import YOLO

if __name__ == "__main__" :
    
    model = YOLO('yolov8s.pt')
    model.train(data="cars.yaml", epochs=100, batch=16, lrf=0.025, device='mps')