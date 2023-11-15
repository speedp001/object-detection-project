import os
import glob
import cv2
import xml.etree.ElementTree as ET

from ultralytics import YOLO

# 전에 학습한 모델을 이용해서 test데이터를 input값으로 넣어서 결과값을 예측한 뒤에 그 결과값을 xml형태로 저장하는 코드

# model load
model = YOLO("./glass_project/runs/detect/train/weights/best.pt")
data_path =  "./glass_project/glass_dataset/test"
data_path_list = glob.glob(os.path.join(data_path, "*.jpg"))

tree = ET.ElementTree()
root = ET.Element("annotations")

"""
1.
<annotations>
</annotations>    
"""

id_number = 0
xml_path = "./glass_project/test.xml"

for path in data_path_list :
    names = model.names  # 모델 라벨
    # print(names)
    
    results = model.predict(path, save=False, imgsz=640, conf=0.5)
    # conf: 모델이 얼마나 "자신 있는" 예측 결과를 제시해야 하는지를 결정하는 매개변수
    # imgsz: 정사각형으로 640 -> imgsz=(width, height)
    # print(results) # 모델 예측 결과
    
    boxes = results[0].boxes
    box_info = boxes
    # print(box_info)
    """
    boxes: 예측된 객체의 바운딩 박스 좌표와 해당 박스의 신뢰도(confidence) 및 클래스 정보가 포함된 텐서입니다. 각 행은 한 객체에 대한 정보를 나타냅니다.

    cls: 클래스 정보를 담고 있는 텐서입니다. 각 값은 해당 객체가 어떤 클래스에 속하는지를 나타냅니다.

    conf: 예측된 객체의 신뢰도(확률)를 나타내는 텐서입니다. 이 값이 일정 신뢰도 이상이어야 객체로 인식됩니다.

    data: 예측된 객체의 정보가 담긴 텐서입니다. 각 행은 한 객체에 대한 정보를 나타냅니다. 예를 들면, 객체의 속도, 크기, 방향 등의 특성이 여기에 포함될 수 있습니다.

    id: 객체의 식별자를 나타내는 변수로 보이지만 여기서는 None으로 설정되어 있습니다.

    is_track: 객체 추적 여부를 나타내는 변수로 보이지만 여기서는 False로 설정되어 있습니다.

    orig_shape: 원본 이미지의 크기를 나타내는 튜플입니다.

    shape: 결과의 모양(shape)을 나타내는 튜플입니다.

    xywh: 바운딩 박스의 중심 좌표와 너비, 높이를 나타내는 텐서입니다.

    xywhn: 정규화된 바운딩 박스의 중심 좌표와 너비, 높이를 나타내는 텐서입니다.

    xyxy: 바운딩 박스의 좌표를 나타내는 텐서입니다.

    xyxyn: 정규화된 바운딩 박스의 좌표를 나타내는 텐서입니다.
    """
    box_xyxy = box_info.xyxy
    # print(path, box_xyxy)
    cls = box_info.cls
    # print(cls)
   
    image = cv2.imread(path)
    img_height, img_width, _ = image.shape
    file_name = os.path.basename(path)
    # print(file_name, img_height, img_width)
    
    xml_frame = ET.SubElement(root, "image", id="%d" % id_number, name=file_name, width="%d" % img_width, height="%d" % img_height)
    
    """
    2.
    <annotations>
        <image id="0" name="IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg", width="1920" height="1080">
    </annotations>  
    """
    
    for bbox, class_number in zip(box_xyxy, cls) :
        class_number = int(class_number.item())
        class_name_temp = names[class_number] # 0을 통해 glass반환
        print(class_name_temp)
        """
          3. 
          <annotations>
              <image id="0" name="IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg" width="1920" height="1080">
              <box label="glass" source="manual" occluded="0" xtl="1026.81" ytl="324.65" xbr="1309.74" ybr="479.46" z_order="0"> </box>
          </annotations>
         """
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())
        ET.SubElement(xml_frame, "box", label=str(class_name_temp), source="manual",occluded="0",
                      xtl=str(x1), ytl=str(y1), xbr=str(x2), ybr=str(y2), z_order="0")

    id_number +=1
    tree._setroot(root)
    tree.write(xml_path, encoding="utf-8")