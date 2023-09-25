import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class JIGSAWS(Dataset):
    def __init__(self, data_root, is_train, tasks=['Knot_Tying', 'Needle_Passing', 'Suturing'], postfix=[0]):
        self.data_root = data_root
        
        if is_train:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        # self.preprocess_fn = preprocess_fn

        tasks = [x.split('_')[0] for x in tasks]
        self.img_list = [os.path.join(data_root, x) for x in os.listdir(data_root) if x.endswith('.jpg') and int(x[-5]) in postfix and x.split('_')[0] in tasks]
        # self.mask_list = [os.path.join(data_root, x) for x in os.listdir(data_root) if x.endswith('.npy') and int(x[-5]) in postfix and x.split('_')[0] in tasks]
        print(f"number of images: {len(self.img_list)}")
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = img_path.replace('.jpg', '.npy')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        # image = np.transpose(image, (2, 0, 1))
        # image = torch.tensor(image).float()

        mask = F.one_hot(mask.long(), num_classes=3)
        mask = mask.permute(2, 0, 1)

        return image, mask
    

if __name__ == "__main__":
    data = JIGSAWS('../segment_anything', is_train=True, tasks=['Knot_Tying', 'Needle_Passing'], postfix=list(range(10)))
    img, mask = data[0]
    print(img.shape, mask.shape)