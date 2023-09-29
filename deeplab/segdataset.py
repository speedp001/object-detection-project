import cv2
import os
import numpy as np

from PIL import Image
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset


class customVOCSegmentation(VOCSegmentation):
    def __init__(self, root, mode="train", transforms=None):
        self.root = root
        
        # 상속받은 클래스의 인스턴스 변수에 접근
        super().__init__(root=self.root, image_set=mode, 
                         download=self.check_if_path_exists(), transforms=transforms)
        # VOCSegmentation.__init__()
        # VOCSegmentation 클래스의 생성자를 호출하며, 변경이 필요한 인자에 대해 받아서 넘겨줌
        # 해당 생성자에서 Custom Dataset의 생성자에서 필요한 작동을 모두 정의하고 있음
        # 즉, 데이터셋 자체와 라벨 이미지 모두가 이 단락에서 생성
    
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        mask = np.array(Image.open(self.masks[idx]))
        # 부모 클래스에서 원본 이미지는 self.images라는 이름으로 호출하도록 정의되어있음
        # 라벨에 해당하는 마스크는 self.masks라는 이름으로 호출하도록 정의되어있음
        # 해당 리스트는 이미지 자체가 아닌 해당 이미지 파일 경로만을 담고 있기 때문에, imread로 읽어줘야 함

        # 이 때, Color map으로 되어있는 원본 이미지를 0~20에 해당하는 값으로 변환하기 위해서
        # PIL Image로 읽은 다음, 해당 PIL Image 객체를 np.array로 변환하여 호출
        # (Pillow Image - cv2 image 사이의 Color map 차이를 이용하여 class mapping이 이루어짐)
        
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            # cv2를 최대한 사용하는 것을 가정하였으므로, torchvision의 transform이 아닌
            # albumentations 모듈을 사용하는 형태로 작성
            img = augmented['image']
            mask = augmented['mask']
            # mask[mask > 20] = 0

        return img, mask
    
    def check_if_path_exists(self):
        return False if os.path.exists(self.root) else True
        # self.root에 해당하는 data 폴더는 원래는 만들어지지 않았다가 
        # 클래스 선언과 함게 다운로드 받으며 만들어질 것
        
    
    # __len__의 경우, 부모 클래스에서 자신의 구조를 정의하면서 함께 정의를 한 상태이기 때문에 재정의가 필요 없음
            
            
if __name__ == "__main__":
    
    dataset = customVOCSegmentation("./deeplab/dataset")
    
    for item in dataset:
        img, mask = item
        
        # mask 이미지의 크기를 img 이미지와 동일하게 변경 -> 채널 수를 변경
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        summary = cv2.copyTo(img, mask)
        marked = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        # cv2.imshow("org", img)
        # cv2.imshow("org", mask)
        
        cv2.imshow("summary", summary)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            exit()
