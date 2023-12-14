from ultralytics import YOLO

if __name__ == "__main__" :
    
    model = YOLO("yolov8m.pt")
    
    model.train(data="swimmer.yaml", epochs=100, batch=8, hsv_s=0, hsv_v=0, degrees=5, lrf=0.0025, device="mps")
    
    # simmer.yaml 파일에 학습경로와 class label이 지정되어있다
    # hsv_s 인자는 HSV 색 공간에서 색조(saturation)의 변화량을 지정하는 인자입니다. hsv_s=0으로 지정하면, 색조의 변화가 없습니다
    # hsv_v 인자는 HSV 색 공간에서 명도(value)의 변화량을 지정하는 인자입니다. hsv_v=0으로 지정하면, 명도의 변화가 없습니다
    # degrees 인자는 이미지 회전 각도 범위를 지정하는 인자입니다. degrees=5으로 지정하면, 이미지를 5도 내외로 회전시켜 학습합니다
    # lrf 인자는 학습률을 지정하는 인자입니다. 학습률은 가중치와 편향을 업데이트하는 양을 의미합니다. lrf=0.0025으로 지정하면, 학습률을 0.0025로 설정합니다