import cv2

def draw_boxes_on_image(image_file, annotation_file) :
    # image load
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # txt file read
    with open(annotation_file, 'r', encoding='utf-8') as f :
        lines = f.readlines()

    for line in lines :
        values = list(map(float, line.strip().split(' ')))
        # values [20.0, 1084.0, 395.0, 1395.0, 395.0, 1395.0, 688.0, 1084.0, 688.0]
        # map(float, )는 float형태로 리스트 생성
        # strip은 문자열의 앙끝 공백을 제거
        # 문자열을 공백을 기준으로 split
        class_id = int(values[0])
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max, y_max = int(round(max(values[3], values[5], values[7]))),\
            int(round(max(values[4], values[6], values[8])))

        """
        첫 번째 포인트 (x1, y1): 좌측 상단 좌표를 의미합니다. (x1, y1)은 바운딩 박스의 좌측 상단 꼭지점의 x, y 좌표를 나타냅니다.

        두 번째 포인트 (x2, y2): 우측 상단 좌표를 의미합니다. (x2, y2)은 바운딩 박스의 우측 상단 꼭지점의 x, y 좌표를 나타냅니다.

        세 번째 포인트 (x3, y3): 우측 하단 좌표를 의미합니다. (x3, y3)은 바운딩 박스의 우측 하단 꼭지점의 x, y 좌표를 나타냅니다.

        네 번째 포인트 (x4, y4): 좌측 하단 좌표를 의미합니다. (x4, y4)은 바운딩 박스의 좌측 하단 꼭지점의 x, y 좌표를 나타냅니다.
        """
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min -5), cv2.FONT_HERSHEY_PLAIN, 5,
                    (0,255,0), 2)

    cv2.imshow("test", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()


if __name__ == "__main__" :
    # folder path
    image_file = "./cars_project/cars_dataset/train/syn_00025.png"
    annotation_file = "./cars_project/cars_dataset/train/syn_00025.txt"
    
    draw_boxes_on_image(image_file, annotation_file)