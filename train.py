import os
import numpy as np
import random
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import monai

from dataset import JIGSAWS
from estimator import PoseEstimator
from render import JIGSAWSRenderer


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, model, trainset, validset):
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # validloader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size)

    jrenderer = JIGSAWSRenderer(args.obj_path, device=args.device)

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr)
    rdice_criterion = monai.losses.DiceLoss(sigmoid=False, squared_pred=True, reduction="mean")
    sdice_criterion = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # bce_criterion = torch.nn.BCELoss(reduction="mean")

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(args.save_path, f'{args.task_name}_{run_id}')
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(f'./log/{args.task_name}_{run_id}')
    
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    device = args.device
    model = model.to(device)
    # model = torch.compile(model)

    epochs = args.epochs
    best_dice = 0

    for e in range(start_epoch, epochs):
        with tqdm(desc=f'Train Epoch {e+1}', total=len(trainloader)) as t:
            train_loss = 0
            train_rdice = 0
            train_sdice = 0
            train_batch = 0

            model.train()
            for i, (image, mask) in enumerate(trainloader):
                image, mask = image.to(device), mask.to(device)
                optimizer.zero_grad()

                if args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        x, m = model(image)
                        rendered = jrenderer.render_batch_masks(x)
                        render_dice_loss = rdice_criterion(rendered[:, 1:], mask[:, 1:])
                        seg_dice_loss = sdice_criterion(m[:, 1:], mask[:, 1:])
                        loss = render_dice_loss + seg_dice_loss
                    
                    if torch.isnan(loss).sum() > 0:
                        print(torch.isnan(image).sum(), torch.isnan(mask).sum())
                        print(render_dice_loss, seg_dice_loss)
                        continue
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                else:
                    x, m = model(image)
                    rendered = jrenderer.render_batch_masks(x)
                    render_dice_loss = rdice_criterion(rendered[:, 1:], mask[:, 1:])
                    seg_dice_loss = sdice_criterion(m[:, 1:], mask[:, 1:])
                    loss = render_dice_loss + seg_dice_loss

                    if torch.isnan(loss).sum() > 0:
                        print(torch.isnan(image).sum(), torch.isnan(mask).sum())
                        print(render_dice_loss, seg_dice_loss)
                        continue
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                batch = image.size(0)
                train_loss += loss.item() * batch
                train_rdice += (1 - render_dice_loss.item()) * batch
                train_sdice += (1 - seg_dice_loss.item()) * batch
                train_batch += batch

                t.set_postfix({'Train Loss': f'{train_loss / train_batch:.4f}',
                               'Train DiceR': f'{train_rdice / train_batch:.4f}', 
                               'Train DiceS': f'{train_sdice / train_batch:.4f}'})
                t.update(1)

            writer.add_scalar('Loss/train', train_loss / train_batch, e+1)
            writer.add_scalar('Render_Dice/train', train_rdice / train_batch, e+1)
            writer.add_scalar('Seg_Dice/train', train_sdice / train_batch, e+1)

        with tqdm(desc=f'Valid Epoch {e+1}', total=len(validloader)) as t:
            valid_loss = 0
            valid_rdice = 0
            valid_sdice = 0
            valid_batch = 0

            model.eval()
            with torch.no_grad():
                for i, (image, mask) in enumerate(validloader):
                    image, mask = image.to(device), mask.to(device)

                    if args.use_amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            x, m = model(image)
                            rendered = jrenderer.render_batch_masks(x)
                            render_dice_loss = rdice_criterion(rendered[:, 1:], mask[:, 1:])
                            seg_dice_loss = sdice_criterion(m[:, 1:], mask[:, 1:])
                            loss = render_dice_loss + seg_dice_loss
                    
                    else:
                        x, m = model(image)
                        rendered = jrenderer.render_batch_masks(x)
                        render_dice_loss = rdice_criterion(rendered[:, 1:], mask[:, 1:])
                        seg_dice_loss = sdice_criterion(m[:, 1:], mask[:, 1:])
                        loss = render_dice_loss + seg_dice_loss

                    batch = image.size(0)
                    valid_loss += loss.item() * batch
                    valid_rdice += (1 - render_dice_loss.item()) * batch
                    valid_sdice += (1 - seg_dice_loss.item()) * batch
                    valid_batch += batch

                    t.set_postfix({'Valid Loss': f'{valid_loss / valid_batch:.4f}', 'Valid DiceR': f'{valid_rdice / valid_batch:.4f}', 'Valid DiceS': f'{valid_sdice / valid_batch:.4f}'})
                    t.update(1)

                writer.add_scalar('Loss/valid', valid_loss / valid_batch, e+1)
                writer.add_scalar('Render_Dice/valid', valid_rdice / valid_batch, e+1)
                writer.add_scalar('Seg_Dice/valid', valid_sdice / valid_batch, e+1)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": e,
        }
        
        torch.save(checkpoint, os.path.join(save_path, f"{args.task_name}_model_latest.pth"))
        if valid_rdice / valid_batch > best_dice:
            best_dice = valid_rdice / valid_batch
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
            }
            torch.save(checkpoint, os.path.join(save_path, f"{args.task_name}_model_best.pth"))


def main(args):
    seed_torch(3407)

    trainset = JIGSAWS(args.data_root, is_train=True, tasks=['Knot_Tying', 'Needle_Passing'], postfix=list(range(10)))
    validset = JIGSAWS(args.data_root, is_train=False, tasks=['Suturing'], postfix=list(range(0, 10, 2)))

    model = PoseEstimator(args.backbone, args.seg_ckpt, dim_feat=2048)
    train(args, model, trainset, validset)


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--task_name', type=str, default='DR2')
    args.add_argument('--backbone', type=str, default='resnet50')

    args.add_argument('--data_root', type=str, default='../segment_anything')
    args.add_argument('--seg_ckpt', type=str, default='')
    args.add_argument('--obj_path', type=str, default='')
    args.add_argument('--save_path', type=str, default='./checkpoints')
    args.add_argument('--resume', type=str, default=None)

    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--lr', type=float, default=3e-4)

    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--num_workers', type=int, default=2)
    
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--use_amp', action='store_true')

    args = args.parse_args()

    args.seg_ckpt = 'checkpoints/DeepLabV3Plus_resnet50_20230925_0213/DeepLabV3Plus_resnet50_model_best.pth'
    args.obj_path = 'GasCylinder02.obj'
    # args.task_name = f'{args.task_name}_{args.backbone}'

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
