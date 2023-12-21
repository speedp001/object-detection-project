import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the model
# 미리 데이터셋으로 수영선수를 학습한 pt파일을 활용
model = YOLO("./swimmer_project/runs/detect/train/weights/best.pt")

# Open the Video file
video_path = "./swimmer_project/ultralytics/swimming_pool.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda : [])

while cap.isOpened() :
    # Read a frame from video
    success, frame = cap.read()
    # print(frame)
    
    if success :
        # Run yolov8 tracking on the frame, persisting tracks between frames
        
        results = model.track(frame, persist=True)
        """
        frame: 추적할 대상이 포함된 현재 프레임 이미지입니다.
        persist: 추적 정보를 이전 프레임에서부터 유지하고 업데이트할지 여부를 지정하는 매개변수입니다. True로 설정하면 추적 결과가 이전 프레임에서의 추적 결과를 기반으로 업데이트되며,
        False로 설정하면 매 프레임마다 새로운 추적이 시작됩니다.
        """
        # print(results)
        
        boxes = results[0].boxes.xyxy.cpu().tolist()
        # print(boxes)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # print(track_ids)
        
        #Draw box by using yolo function plot()
        annotated_frame = results[0].plot()
        
        for box, track_id in zip(boxes, track_ids) :
            
            x1, y1, x2, y2 = box
            track = track_history[track_id]
            track.append((float(x1), float(y1)))
            # 트랙 히스토리를 유지할 때, 상단 왼쪽 모서리 (x1, y1)의 좌표만 사용하여 객체의 이동 경로를 기록합니다.
            # 이전 프레임과 현재 프레임 사이의 이동 및 추적을 기록하는 데에는 (x1, y1) 좌표만으로도 충분합니다.
            
            if len(track) > 30 :
                track.pop(0)  # Retain tracks for the last 30 frames
                
            # Draw the tracking lines
            
            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            # np.hstack(track)을 실행하면 각 (xi, yi) 좌표 쌍을 수평으로 결합하여 아래와 같은 배열이 생성됩니다.
            # array([x1, y1, x2, y2, x3, y3, ...])
            # reshape 과정을 통해서
            """
            points([
                [[x1, y1]],
                [[x2, y2]],
                [[x3, y3]],
                ...
            ])
            """
            
            # 객체의 이동 추적을 나타내는 곡선을 그리는 코드
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(100, 150, 130), thickness=5)

            
        # Display the annotated frame
        cv2.imshow("Tracking...", annotated_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q') :
            break