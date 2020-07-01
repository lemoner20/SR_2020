#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform1 = transform[0]
        self.transform2 = transform[1]
        self.transform3 = transform[2]
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 30000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return three image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        # Return: HR images, LR images, Input images, Selected attribute labels
        return self.transform1(image), self.transform2(image), self.transform3(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, image_size=128, magnification=4,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""

    # Transform1, Ground truth HR images to Tensor
    transform1 = []
    transform1.append(T.ToTensor())
    transform1.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform1 = T.Compose(transform1)

    # Transform2, LR images to Tensor
    transform2 = []
    transform2.append(T.Resize(image_size // magnification, interpolation=Image.BICUBIC))
    transform2.append(T.Resize(image_size, interpolation=Image.NEAREST))
    transform2.append(T.ToTensor())
    transform2.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform2 = T.Compose(transform2)

    # Transform3, HR ->downsample-> LR ->upsample-> Input images to Tensor
    transform3 = []
    # if mode == 'train':
    #     transform3.append(T.RandomHorizontalFlip())
    transform3.append(T.Resize(image_size // magnification, interpolation=Image.BICUBIC))
    transform3.append(T.Resize(image_size, interpolation=Image.BICUBIC))
    transform3.append(T.ToTensor())
    transform3.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform3 = T.Compose(transform3)

    # Transform
    transform = [transform1, transform2, transform3]

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
