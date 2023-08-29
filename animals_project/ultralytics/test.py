import torch
import os
import glob
import cv2

model = torch.hub.load("ultralytics/yolov5", 'custom', path="./animals_project/runs/train/exp/weights/best.pt", force_reload=True)

# 디바이스 설정
device = torch.device("mps")
# MPS 사용 시 아래 환경 변수 설정
torch.backends.mps = True
torch.backends.cudnn.enabled = False

# 모델을 디바이스로 보내기
model.to(device)

model.conf = 0.5
model.iou = 0.45

# # 이미지 한개로 Object Detection 확인하는 test 코드
# img = "./animals_project/animals_dataset/test/images/9_jpg.rf.bca7063b1ce4c9c9b8f10c2fdfd6736d.jpg"
# image = cv2.imread(img)

# label_dict = {
#     0 : "cat",
#     1 : "chicken",
#     2 : "cow",
#     3 : "dog",
#     4 : "fox",
#     5 : "goat",
#     6 : "horse",
#     7 : "person",
#     8 : "racoon",
#     9 : "skunk"
# }

# result = model(img, size=640)
# bboxes = result.xyxy[0]
# print(bboxes)

# for bbox in bboxes :
#     x1, y1, x2, y2, conf, cls = bbox
#     x1 = int(x1.item())
#     y1 = int(y1.item())
#     x2 = int(x2.item())
#     y2 = int(y2.item())
#     conf = conf.item()
#     cls = int(cls.item())
#     cls_name = label_dict[cls]
#     print(x1, y1, x2, y2, conf, cls_name)
#     image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

# cv2.imshow("test", image)
# if cv2.waitKey(0) & 0xFF == ord('q') :
#     exit()

label_dict = {
    0 : "cat",
    1 : "chicken",
    2 : "cow",
    3 : "dog",
    4 : "fox",
    5 : "goat",
    6 : "horse",
    7 : "person",
    8 : "racoon",
    9 : "skunk"
}

img_folder_path = "./animals_project/ultralytics/animals_dataset/test"
img_path = glob.glob(os.path.join(img_folder_path, "*", "*.jpg"))

for path in img_path:
    # print(path)
    image = cv2.imread(path)
    results = model(path, size=640)
    
    bboxes = results.xyxy[0]
    for bbox in bboxes:
        # print(bbox)
        x1, y1, x2, y2, conf, cls = bbox
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        conf = conf.item()
        cls = int(cls.item())
        cls_name = label_dict[cls]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        print(x1, y1, x2, y2, cls_name)
        
    cv2.imshow("test", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()













