import torch
import numpy as np
import time
import os
from tqdm import tqdm

#학습 클래스
class SegLearner:
    def __init__(self, model, optimizer, criterion, train_dataloader, valid_dataloader, args):
        self.device = torch.device("mps")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        # 모델, 손실함수, 옵티마이저
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # 데이터로더
        
        self.args = args
        # 터미널 인자로 받은 필요값들
        # argument 전체를 args에 할당
        
        self.start_epoch = 0
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "train_miou": [],
            "val_loss": [],
            "val_acc": [],
            "val_miou": []
        }
        # resume이 걸릴 경우 / 학습이 저장될 경우 필요한 값들
        
    def train(self):
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            train_loss = 0.0
            train_corr = 0.0
            val_loss = 0.0
            val_corr = 0.0
            
            train_iou = 0.0
            val_iou = 0.0
            # mIoU 계산을 위한 IoU 총합치 저장 변수
            
            for i, (inputs, labels) in enumerate(tqdm(self.train_dataloader)):
                # 이미지
                inputs = inputs.float().to(self.device)
                # print("input", inputs)
                # 정답지 (segementation image)
                labels = labels.long().to(self.device)
                # print("label", labels)
                """
                여기서 inputs는 4차원 labels는 3차원이다. 이유는 labels에서는 배경과 객체 이 두개의 클래스만 구분한다면 클래스 값이 0또는 1로 구분될 수 있기 때문에 차원을 늘릴 필요가 없다.
                0이외의 숫자에서는 숫자를 부여하여 각 클래스의 번호를 새긴다.
                
                deeplab 모델에서는 0~20번호의 숫자가 클래스 개수이고 255는 배경을 의미한다.
                """
                
                self.optimizer.zero_grad()  # 가중치 업데이트를 위한 optimizer 초기화
                
                outputs = self.model(inputs)  # 순전파

                outputs = outputs["out"]  # deeplab은 output이 dict 형태로 쓸 수 있도록 나오므로, 출력치 key로 받아옴
                # 입력 이미지가 520x520 픽셀이면 "out"의 형태는 (배치 크기, 클래스 수, 520, 520)
                
                # # outputs 형태 확인
                # print(outputs.shape)
                
                # # 텐서를 넘파이 배열로 변환
                # outputs_numpy = outputs.cpu().detach().numpy()

                # # 넘파이 배열의 내용 확인
                # print(outputs_numpy)
    
                # 손실 함수
                # PyTorch는 브로드캐스팅(Broadcasting)을 사용하여 손실 함수를 계산하려고 시도하고, 일치하지 않는 차원을 알맞게 확장합니다.
                loss = self.criterion(outputs, labels)
                loss.backward()  # 역전파
                self.optimizer.step()  # 가중치 없데이트
                
                preds = torch.argmax(outputs, dim=1)  # output으로부터 class에 대한 예측값을 얻음
                # dim=1은 텐서의 두 번째 차원(클래스 수에 해당하는 차원)을 따라 최댓값을 찾으라는 의미(클래스 수) -> 가장 높은 클래스의 확률의 인덱스를 선택하여 3차원으로 준다
                
                train_loss += loss.item()
                corrects = torch.sum(preds == labels.data)  # labels와 preds는 이미지 형태
                # 이미지 형태이므로, 여기에서 나오는 총 길이는 최대 520x520 크기(pretrained 기준)
                # 두 이미지를 겹쳐서 일치하는 픽셀 개수를 corrects에 저장
                
                # 총 픽셀개수로 나눠줘서 맞은 비율(정확도)를 구한다
                batch_size = inputs.size(0)
                train_corr = corrects / (batch_size * 520 * 520)
                # 앞에 2를 곱하는 이유는 for문 한번에 batch_size만큼 들어가기 때문에 총 픽셀값에 batch_size를 곱한다. (여기서 batch_size = 2)
                
                train_iou += self.calc_iou(preds, labels.data)
            
            _t_loss = train_loss / len(self.train_dataloader)
            # train_dataloader는 총 데이터셋을 배치사이즈로 나눈 크기
            _t_acc = train_corr / len(self.train_dataloader.dataset)
            # train_dataloader.dataset는 데이터셋의 크기
            _t_miou = train_iou / len(self.train_dataloader.dataset)
            
            self.metrics["train_loss"].append(_t_loss)
            self.metrics["train_acc"].append(_t_acc)
            self.metrics["train_miou"].append(_t_miou)
            
            print(f"[{epoch + 1} / {self.args.epochs}] train loss : {_t_loss}",
                    f"train acc : {_t_acc}, train mIoU : {_t_miou}")
        
            
            # validation 시작
            self.model.eval()
            with torch.no_grad():
                for val_i, (inputs, labels) in enumerate(tqdm(self.valid_dataloader)):
                    inputs = inputs.float().to(self.device)
                    labels = labels.long().to(self.device)
                    
                    outputs = self.model(inputs)
                    outputs = outputs["out"]
                    loss = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    
                    val_loss += loss.item()
                    corrects = torch.sum(preds == labels.data)
                    
                    batch_size = inputs.size(0)
                    val_corr += corrects / (batch_size * 520 * 520)
                    # Pixel accuracy를 이용한 정확도 계산
                    
                    val_iou += self.calc_iou(preds, labels.data)
                    
            _v_loss = val_loss / len(self.valid_dataloader)
            _v_acc = val_corr / len(self.valid_dataloader.dataset)
            _v_miou = val_iou / len(self.valid_dataloader.dataset)
            
            self.metrics["val_loss"].append(_v_loss)
            self.metrics["val_acc"].append(_v_acc)
            self.metrics["val_miou"].append(_v_miou)
            
            print(f"[{epoch + 1} / {self.args.epochs}] train loss : {_v_loss}",
                    f"train acc : {_v_acc}, train mIoU : {_v_miou}")
                    
            self.save_ckpts(epoch)
    
           
    def load_ckpts(self):
        # path: .pt 파일이 저장된 위치
        ckpt_path = os.path.join(self.args.weight_folder, self.args.weight_file_name)
        
        # 터미널 인자 args로부터 기본값으로 지정된 weight 로딩 경로를 받아옴
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path) # .pt 파일을 불러와서 dictionary 형태로 선언
            self.model.load_state_dict(ckpt["model"]) # dict 안에 있는 "model"키로 저장할 모델 가중치 로드
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt["epoch"]
            self.metrics = ckpt["metrics"]
            print(f"Loaded checkpoint from epoch {self.start_epoch}")
        
        else:
            print("Checkpoint file not found. Starting from epoch 0.")

        
        
        
    
    def save_ckpts(self, epoch, file_name=None):
        # 체크포인트 저장을 처리하기 위한 편의함수
        if not os.path.exists(self.args.weight_file_name):
            os.makedirs(self.args.weight_folder_path, exist_ok=True)
        # 모델 가중치가 저장될 폴더가 없을 경우, 오류가 날 수 있으므로
        # 터미널 인자 args에서 받은 model_folder_path가 있는지 확인 후, 없으면 생성

        if file_name is None:
            to_save_path = os.path.join(self.args.weight_folder_path, self.args.weight_file_name)
        else:
            to_save_path = os.path.join(self.args.weight_folder_path, file_name)
        # file name 커스텀을 위한 조건식 부분

        torch.save(
            {
                "model": self.model.state_dict(), # 현재 가중치 값
                "optimizer": self.optimizer.state_dict(), # optimizer의 현재 수치
                "epoch": epoch,
                "metrics": self.metrics
            }, to_save_path
        )
        
        
    @staticmethod
    # 영역 대비 정답률 계산 함수
    def calc_iou(preds, labels):
        
        total_iou = 0.0
        
        for inp, ans in zip(preds, labels):
            # inp == preds에서 들어온 예측치가 담겨있는 단일 이미지(텐서)
            # ans == labels에서 들어온 정답치가 담겨있는 단일 이미지(텐서)
            
            inp = inp.cpu().numpy()
            # inp가 아직 디바이스안에 담겨있기 때문에 cpu()로 넘겨주어서 numpy로 변환한다.
            ans = ans.cpu().numpy()
            
            union_section = np.logical_or(inp, ans)
            inter_section = np.logical_and(inp, ans)
            # 위에서 or, and 연산을 통해 얻은 numpy 행렬은 행렬이며, 계산에 사용할 수 있는 값이 아님
            
            uni_sum = np.sum(union_section)
            inter_sum = np.sum(inter_section)
            # 해당하는 행렬의 총 픽셀 수 (T/F Boolean mask 형태로 나올 것이므로 sum을 하면 픽셀 수를 얻음)
            # == 해당 영역의 넓이
            iou = inter_sum / uni_sum
            # 교집합 넓이 / 합집합 넓이 = IoU
            
            total_iou += inter_sum / uni_sum
            
        '''
        total_iou 변수는 단일 이미지의 IoU 값을 나타내는 것이 아니라, 배치에 있는 모든 이미지의 IoU 값을 합한 값입니다
        배치에 대한 평균 IoU 값을 얻으려면 total_iou를 배치에 있는 이미지 수로 나누어야 합니다.
        '''
        return total_iou
            