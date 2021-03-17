#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

from utils import load_hdr_as_tensor

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_supersample_dataset(root_dir, target_dir, redux, params, shuffled=False, single=False):
    dataset = SuperSampleDataset(root_dir, target_dir, redux)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs

    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')

    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class SuperSampleDataset(Dataset):

    def __init__(self, root_dir, target_dir, redux):
        super(SuperSampleDataset, self).__init__()

        self.imgs = []
        self.targ_imgs = []
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.redux = redux

        # check here if input and target images names are the same!!!
        self.imgs = os.listdir(root_dir)
        self.targ_imgs = os.listdir(target_dir)

        if redux:
            self.imgs = self.imgs[:redux]
            self.targ_imgs = self.targ_imgs[:redux]

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        source = tvF.to_tensor(img)

        targ_path = os.path.join(self.target_dir, self.targ_imgs[index])
        targ = Image.open(targ_path).convert('RGB')
        target = tvF.to_tensor(targ)

        return source, target

    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)



