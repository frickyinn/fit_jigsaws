import os
from tqdm import tqdm
from datetime import datetime
import argparse

import monai
import segmentation_models_pytorch as smp

import torch
from torch.utils.data import DataLoader

from dataset import JIGSAWS


def train(args, model, trainset, validset):
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # validloader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr)
    dice_criterion = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(args.save_path, f'{args.task_name}_{run_id}')
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    device = args.device
    model = model.to(device)
    # model = torch.compile(model)

    epochs = args.epochs
    best_dice = 0

    for e in range(epochs):
        with tqdm(desc=f'Train Epoch {e+1}', total=len(trainloader)) as t:
            train_loss = 0
            train_dice = 0
            train_batch = 0

            model.train()
            for i, (image, mask) in enumerate(trainloader):
                image, mask = image.to(device), mask.to(device)
                optimizer.zero_grad()

                if args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(image)
                        dice_loss = dice_criterion(pred, mask)
                        loss = dice_loss + bce_criterion(pred, mask.float())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                else:
                    pred = model(image)
                    dice_loss = dice_criterion(pred, mask)
                    loss = dice_loss + bce_criterion(pred, mask.float())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                batch = image.size(0)
                train_loss += loss.item() * batch
                train_dice += (1 - dice_loss.item()) * batch
                train_batch += batch

                t.set_postfix({'Train Loss': f'{train_loss / train_batch:.4f}', 'Train Dice': f'{train_dice / train_batch:.4f}'})
                t.update(1)

        with tqdm(desc=f'Valid Epoch {e+1}', total=len(validloader)) as t:
            valid_loss = 0
            valid_dice = 0
            valid_batch = 0

            model.eval()
            with torch.no_grad():
                for i, (image, mask) in enumerate(validloader):
                    image, mask = image.to(device), mask.to(device)

                    if args.use_amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            pred = model(image)
                            dice_loss = dice_criterion(pred, mask)
                            loss = dice_loss + bce_criterion(pred, mask.float())
                    
                    else:
                        pred = model(image)
                        dice_loss = dice_criterion(pred, mask)
                        loss = dice_loss + bce_criterion(pred, mask.float())

                    batch = image.size(0)
                    valid_loss += loss.item() * batch
                    valid_dice += (1 - dice_loss.item()) * batch
                    valid_batch += batch

                    t.set_postfix({'Valid Loss': f'{valid_loss / valid_batch:.4f}', 'Valid Dice': f'{valid_dice / valid_batch:.4f}'})
                    t.update(1)


        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": e,
        }
        torch.save(checkpoint, os.path.join(save_path, f"{args.task_name}_model_latest.pth"))
        ## save the best model
        if valid_dice / valid_batch > best_dice:
            best_dice = valid_dice / valid_batch
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
            }
            torch.save(checkpoint, os.path.join(save_path, f"{args.task_name}_model_best.pth"))


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--task_name', type=str, default='DeepLabV3Plus')
    args.add_argument('--backbone', type=str, default='resnet50')
    args.add_argument('--data_root', type=str, default='../segment_anything')
    args.add_argument('--save_path', type=str, default='./checkpoints')

    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--lr', type=float, default=3e-4)

    args.add_argument('--batch_size', type=int, default=32)
    # args.add_argument('--num_workers', type=int, default=2)
    
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--use_amp', action='store_false')

    args = args.parse_args()
    args.task_name = f'{args.task_name}_{args.backbone}'

    return args


def main(args):
    trainset = JIGSAWS(args.data_root, is_train=True, tasks=['Knot_Tying', 'Needle_Passing'], postfix=list(range(10)))
    validset = JIGSAWS(args.data_root, is_train=False, tasks=['Suturing'], postfix=list(range(0, 10, 2)))
    model = smp.DeepLabV3Plus(args.backbone, encoder_weights='imagenet', classes=3)
    # model.load_state_dict(torch.load('checkpoints/DeepLabV3Plus_resnet50_20230920_0846/DeepLabV3Plus_resnet50_model_best.pth')['model'])
    
    train(args, model, trainset, validset)


if __name__ == "__main__":
    args = get_args()
    main(args)
