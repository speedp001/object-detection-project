from ultralytics import YOLO

# train model info
model = YOLO('yolov8s.pt')

if __name__ == '__main__':
    model.train(data="flowchart.yaml", epochs=100, batch=16, degrees=5, lrf=0.025, device="mps")