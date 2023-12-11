import json
import os
import shutil

"""
out label -> 0,1,7 제외시키는 라벨 id
in label -> 2,3,4,5,6 사용하는 라벨 id

"categories": [
    {
        "id": 0,
        "name": "aerial-pool",
        "supercategory": "none"
    },
    {
        "id": 1,
        "name": "black-hat",
        "supercategory": "aerial-pool"
    },
    {
        "id": 2,
        "name": "body",
        "supercategory": "aerial-pool"
    },
    {
        "id": 3,
        "name": "bodysurface",
        "supercategory": "aerial-pool"
    },
    {
        "id": 4,
        "name": "bodyunder",
        "supercategory": "aerial-pool"
    },
    {
        "id": 5,
        "name": "swimmer",
        "supercategory": "aerial-pool"
    },
    {
        "id": 6,
        "name": "umpire",
        "supercategory": "aerial-pool"
    },
    {
        "id": 7,
        "name": "white-hat",
        "supercategory": "aerial-pool"
    }
],
"""

def convert_coco_to_yaml(image_path, yolo_image_copy_path, coco_annotations_path, yolo_annotations_path) :
    
    # coco annotations 로드
    with open(coco_annotations_path, 'r', encoding='utf-8') as f :    
        coco_annotation_info = json.load(f)


    image_infos = coco_annotation_info['images']
    anno_infos = coco_annotation_info['annotations']

    for image_info in image_infos :
        image_file_name = image_info['file_name']
        file_name = image_file_name.replace(".jpg", "")
        id = image_info['id']
        image_width = image_info["width"]
        image_height = image_info["height"]
        
        for anno_info in anno_infos :
            if anno_info["image_id"] == id :
                
                category_id = anno_info["category_id"]
                
                """
                out label -> 0,1,7 제외시키는 라벨 id
                in label -> 2,3,4,5,6 사용하는 라벨 id -> swimmer 0
                """
                
                # 라벨 값이 0, 1, 7을 제외한 부분만 고려
                if category_id not in [0, 1, 7] :
                    
                    x,y,w,h = anno_info['bbox']
                    
                    # x, y, w, h -> center_x, center_y, w, h
                    x_center = (x + w/ 2) / image_width
                    y_center = (y + h / 2) / image_height
                    w /= image_width
                    h /= image_height
                    
                    # copy swimmer_dataset to yolo swimmer_yolo_dataset
                    source_image_path = os.path.join(image_path, image_file_name)
                    destination_image_path = os.path.join(yolo_image_copy_path, image_file_name)
                    shutil.copy(source_image_path, destination_image_path)
                    
                    # write to text file
                    # 2,3,4,5,6 라벨 값은 모두 0번으로 지정 -> .yaml파일에 0: swimmer라고 지정
                    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                    text_path = os.path.join(yolo_annotations_path, f"{file_name}.txt")
                    
                    with open(text_path, 'a') as f :
                        f.write(yolo_line)
                    
    print("done")
        
    
    

# yolo형식의 데이터 셋 폴더 생성
os.makedirs("./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/train/images", exist_ok=True)
os.makedirs("./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/train/labels", exist_ok=True)
os.makedirs("./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/valid/images", exist_ok=True)
os.makedirs("./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/valid/labels", exist_ok=True)

# 원본 이미지 경로
image_train_path = "./object_tracking/swimmer_dataset/train/"
image_valid_path = "./object_tracking/swimmer_dataset/valid/"

# yolo format images 저장 경로
yolo_image_train_copy_path = "./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/train/images"
yolo_image_valid_copy_path = "./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/valid/images"

# coco형식의 annotation file 경로
coco_annotation_train_path = "./object_tracking/swimmer_dataset/train/_annotations.coco.json"
coco_annotation_valid_path = "./object_tracking/swimmer_dataset/valid/_annotations.coco.json"

# yolo format annotations 저장 경로
yolo_annotation_train_path = "./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/train/labels"
yolo_annotation_valid_path = "./object_tracking/ultralytics/ultralytics/cfg/swimmer_yolo_datset/valid/labels"



convert_coco_to_yaml(image_train_path, yolo_image_train_copy_path, coco_annotation_train_path, yolo_annotation_train_path)
convert_coco_to_yaml(image_valid_path, yolo_image_valid_copy_path, coco_annotation_valid_path, yolo_annotation_valid_path)
                
    
