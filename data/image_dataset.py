import os
import pickle
import logging
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, pickle_file_path, stage='train') -> None:
        super().__init__()
        with open(pickle_file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
        self.img_path          = os.path.join(os.path.dirname(pickle_file_path), 'images')
        self.mask_path         = os.path.join(os.path.dirname(pickle_file_path), 'masks')
        self.img_size          = 224
        self.stage             = stage
        self.name_list         = loaded_dict[stage]['name_list']
        self.transform         = self.get_transforms()
        logging.info('{} set num: {}'.format(stage, len(self.name_list)))

        del loaded_dict

    def get_transforms(self):
        if self.stage == 'train':
            transforms = A.Compose([
                A.ToFloat(max_value=255.0, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.Resize(self.img_size, self.img_size, always_apply=True),
                ToTensorV2(always_apply=True),
            ])
        else:
            transforms = A.Compose([
                A.ToFloat(max_value=255.0, always_apply=True),
                A.Resize(self.img_size, self.img_size, always_apply=True),
                ToTensorV2(always_apply=True),
            ])
        return transforms

    def __getitem__(self, index):
        name = self.name_list[index]
        # load img & seg
        seg_image = Image.open(os.path.join(self.mask_path, name + '.png')).convert("RGB")
        seg_data  = np.array(seg_image).astype(np.float32)
        img_image = Image.open(os.path.join(self.img_path,  name + '.png')).convert("RGB")
        img_data  = np.array(img_image).astype(np.float32)

        augmented = self.transform(image=img_data, mask=seg_data)

        aug_img = augmented['image']
        aug_seg = augmented['mask']

        return {
            'name': name,
            'img': aug_img,
            'seg': aug_seg
        }

    def __len__(self):
        return len(self.name_list)

