import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict



# Load the yolov8 model
# 기존 사람을 학습해논 yolo8m 모델을 사용
model = YOLO("yolov8m.pt")

# open the video
# 추적할 비디오 지정
video_path = "./swimmer_project/ultralytics/track_person_data.webm"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda : [])
# defaultdict을 사용하면 특정 키에 해당하는 값이 없을 때, 기본적으로 지정한 값(여기서는 빈 리스트 [])을 반환합니다.
# key는 객체의 track ID이고, value는 각 프레임에서 객체의 위치를 나타내는 좌표 리스트입니다.

while cap.isOpened() :
    # Read a frame from video
    # success: 프레임을 성공적으로 읽었는지 여부를 나타내는 불리언 값입니다.
    # frame: 현재 프레임 이미지가 포함된 변수입니다.
    success, frame = cap.read()
    # print(frame)
    
    if success :
        # Run yolov8 tracking on the frame, persisting tracks between frames
        
        # 기존 yolov8모델로 학습되어있는 가중치와 편향값을 이용해서 현재 프레임에 인퍼런스 값을 얻는다.
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
        
        # Draw box by using yolo function plot()
        # annotated_frame은 시각적으로 표시된 프레임을 나타내는 이미지
        # results[0].plot()는 Ultralytics YOLO 라이브러리에서 제공되는 메소드로, 객체 추적 결과를 시각화하기 위해 사용
        annotated_frame = results[0].plot()
        
        for box, track_id in zip(boxes, track_ids) :
            
            x1, y1, x2, y2 = box
            track = track_history[track_id]
            track.append((float(x1), float(y1)))
            # 트랙 히스토리를 유지할 때, 상단 왼쪽 모서리 (x1, y1)의 좌표만 사용하여 객체의 이동 경로를 기록합니다.
            # 이전 프레임과 현재 프레임 사이의 이동 및 추적을 기록하는 데에는 (x1, y1) 좌표만으로도 충분합니다.
            
            if len(track) > 30 :
                # track.pop(0): track 리스트의 첫 번째 요소를 삭제합니다. 이렇게 하는 것은 리스트의 길이를 30으로 유지하기 위해 가장 오래된 이동 경로를 제거
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
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(210, 150, 180), thickness=5)
            # cv2.polylines는 여러 개의 다각형을 그릴 수 있도록 설계되어 있습니다. 따라서 여러 개의 다각형을 그릴 때는 다양한 다각형의 좌표 리스트를 담은 리스트를 전달해야 합니다.
            # isClosed=False로 설정하면 열린 곡선을 그린다 
            # color는 곡선의 색
            # thickness는 선의 두께

            
        # Display the annotated frame
        cv2.imshow("Tracking...", annotated_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q') :
            break
            
            
        