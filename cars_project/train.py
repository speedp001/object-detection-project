import warnings
warnings.filterwarnings(action='ignore')
# 경고 메시지를 무시하도록 설정하는 Python의 warnings 모듈의 함수
import random
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
from custaomdataset import CustomDataset, collate_fn
from config import config

def main() :

    device = torch.device('cpu')
    print(f"device : {device}")

    # Fixed Random-seed
    # 함수의 재현성 보장을 위한 설정
    def seed_everything(seed) :
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    seed_everything(config['SEED']) # Seed fix

    # aug
    def get_train_transforms() :
        return A.Compose([
            A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        # format='pascal_voc'은 bounding box 좌표 형식을 Pascal VOC 형식으로 지정하는 것을 의미
        # Pascal VOC 형식은 각 bounding box가 (x_min, y_min, x_max, y_max) 형태로 지정을 의미
        # label_fields=['labels']를 사용하여 레이블 정보를 bounding box와 함께 튜플로 묶어주는 방식으로 사용

    def get_test_transforms() :
        return A.Compose([
            A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
            ToTensorV2()
        ])

    # dataset dataloader
    train_dataset = CustomDataset("./cars_project/cars_dataset/train/" , train=True, transforms=get_train_transforms())
    test_dataset = CustomDataset("./cars_project/cars_dataset/test", train=False, transforms=get_test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    def build_model(num_classes=config['NUM_CLASS'] + 1) :
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        return model

    model = build_model()
    model.to(device)

    def train(model, train_loader, optimizer, scheduler, device, resume_checkpoint = None) :
        best_loss = 999999

        start_epoch = 1

        if resume_checkpoint is not None :
            checkpoint = torch.load(resume_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epcoh'] + 1
            print(f"Resuming training from epoch {start_epoch}")  

        for epoch in (range(start_epoch, config['EPOCHS'] + 1)) :

            model.train()
            train_loss = []
            num_batches = len(train_loader)

            for batch_idx, (images, targets) in enumerate(
                tqdm(train_loader, total=num_batches, desc=f"Epoch [{epoch}] Batches", leave=True, mininterval=0)):


                images = [img.to(device) for img in images]
                targets = [{k:v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                optimizer.step()

                train_loss.append(losses.item())

            tr_loss = np.mean(train_loss)
            tqdm.write(f"Epoch [{epoch}] Train loss : {tr_loss:.5f}")

            if scheduler is not None :
                scheduler.step()

            if best_loss > tr_loss :
                best_loss = tr_loss
                best_model = model.state_dict()
                torch.save(best_model, './cars_project/best.pt')

            # save checkpoint
            checkpoint = {
                'epcoh' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'best_loss' : best_loss
            }
            torch.save(checkpoint, "./cars_project/checkpoint.pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train(model,  train_loader, optimizer, scheduler, device, resume_checkpoint='./cars_project/checkpoint.pt')


if __name__ == "__main__" :
    main()