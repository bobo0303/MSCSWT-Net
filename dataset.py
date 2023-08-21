import os
import random
from glob import glob
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset

class get_dataset(Dataset):
    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, image_size=256):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        self.image_id_list = []
        with open(self.pt_dataset) as f:
            for line in f:
                self.image_id_list.append(line.strip())

        if is_train:
            self.strokes_mask_list = []
            with open(mask_path) as f:
                for line in f:
                    self.strokes_mask_list.append(line.strip())
            self.strokes_mask_list = sorted(self.strokes_mask_list, key=lambda x: x.split('/')[-1])

        else:
            # self.mask_list = glob(test_mask_path + '/*')
            # self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])
            self.test_mask_list = []
            with open(test_mask_path) as f:
                for line in f:
                    self.test_mask_list.append(line.strip())
            self.test_mask_list = sorted(self.test_mask_list, key=lambda x: x.split('/')[-1])

        self.image_size = image_size    #default = 256
        self.training = is_train

    def __len__(self):
        return len(self.image_id_list)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.test_mask_list[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode:  random brush
            mask_index = random.randint(0, len(self.strokes_mask_list) - 1)
            mask = cv2.imread(self.strokes_mask_list[mask_index],
                              cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)
        return img

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name)
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])
        img = img[:, :, ::-1]

        img = self.resize(img, self.image_size, self.image_size, center_crop=False)
        img_gray = rgb2gray(img)
        # edge = self.load_edge(img_gray)
        # load mask
        mask = self.load_mask(img, idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...].copy()
                # edge = edge[:, ::-1].copy()
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()

        img = self.to_tensor(img, norm=False)
        # edge = self.to_tensor(edge)
        mask = self.to_tensor(mask)
        #mask_img = img * (1 - mask)

        # meta = {'img': img, 'mask': mask, 'edge': edge,
        #         'name': os.path.basename(selected_img_name)}
        meta = {'img': img, 'mask': mask,
                'name': os.path.basename(selected_img_name)}
        return meta



