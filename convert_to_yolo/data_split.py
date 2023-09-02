import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# data folder path
image_folder_path = "./convert_to_yolo/org_dataset/images/"
annotation_folder_path = "./convert_to_yolo/org_dataset/annotations"

#new folder
train_folder = "./convert_to_yolo/csv_dataset/train"
eval_folder = "./convert_to_yolo/csv_dataset/val"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)

#csv path
csv_file_path = os.path.join(annotation_folder_path, "annotations.csv")

#csv file -> pandas DataFrame read
annotation_df = pd.read_csv(csv_file_path)

image_names = annotation_df['filename'].unique()  #unique() -> 중복된 파일 이름을 제거하여 고유한 원소로만 배열 정리
train_names, eval_names = train_test_split(image_names, test_size=0.2)
# print(image_names)
print(f"images name len : {len(image_names)}\n")
print(f"train data size : {len(train_names)}\n")
print(f"train data size : {len(eval_names)}\n")

#train data copy and bounding box info save
train_annotations = pd.DataFrame(columns=annotation_df.columns)  #원본 annotations의 틀을 기준으로 생성

for image_name in train_names :
    print("image_name value(train) >> ", image_name)
    
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(train_folder, image_name)
    #print(new_image_path)
    shutil.copy(img_path, new_image_path)
    
    #annotation csv
    annotation = annotation_df.loc[annotation_df['filename'] == image_name].copy()
    annotation['filename'] = image_name
    train_annotations = train_annotations._append(annotation)
    
print(train_annotations)
train_annotations.to_csv(os.path.join(train_folder, "annotations.csv"), index=False)

#eval data copy and bounding box info save
eval_annotations = pd.DataFrame(columns=annotation_df.columns)  #원본 annotations의 틀을 기준으로 생성

for image_name in eval_names :
    print("image_name value(eval) >> ", image_name)
    
    img_path = os.path.join(image_folder_path, image_name)
    new_image_path = os.path.join(eval_folder, image_name)

    shutil.copy(img_path, new_image_path)
    
    #annotation csv
    annotation = annotation_df.loc[annotation_df['filename'] == image_name].copy()
    annotation['filename'] = image_name
    eval_annotations = eval_annotations._append(annotation)
    
print(eval_annotations)
eval_annotations.to_csv(os.path.join(eval_folder, "annotations.csv"), index=False)