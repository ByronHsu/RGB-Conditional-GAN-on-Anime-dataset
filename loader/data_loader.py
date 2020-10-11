import os
import re
import sys
import torch
import logging
import random
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from os.path import join
from .base_data_loader import BaseDataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image 
from utils import embed, rgb
class Dataset(data.Dataset):
    def __init__(self, data_dir, n_emb, method = 'cgan'):
        img_list = os.listdir(join(data_dir, 'images'))
        # sorted as 1.jpg 2.jpg ... 
        img_list.sort(key = lambda x: int(x.split('.')[0]))
        tag_list = []

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # reade tags.csv
        tags = None
        with open(join(data_dir, 'tags.csv'), 'r') as f: 
            tags = f.readlines()

        dic_hair, dic_eyes = self._build_dict(tags)
        # dic_hair, dic_eyes, rgb_hair, rgb_eyes = self._build_dict(tags)
        
        for t in tags:
            attr = re.search(r'\d+,(.*) hair (.*) eyes', t)
            a1, a2 = attr.group(1), attr.group(2)
            tag_list.append((dic_hair[a1], dic_eyes[a2]))
        self.data_dir = data_dir
        self.img_list = img_list
        self.tag_list = tag_list
        self.dic_hair, self.dic_eyes = dic_hair, dic_eyes
        # self.rgb_hair, self.rgb_eyes = rgb_hair, rgb_eyes
        self.n_emb = n_emb
        self.method = method

    def __getitem__(self, index):
        img = self._pil_loader(join(self.data_dir, 'images', self.img_list[index]))
        img = self.transform(img)
        right_h, right_e = self.tag_list[index]
        right_emb = embed(right_h, right_e, n_emb = self.n_emb)
        wrong_h, wrong_e = self._random_butnot(right_h, right_e)
        wrong_emb = embed(wrong_h, wrong_e, n_emb = self.n_emb)
        if self.method == 'cgan' or self.method == 'wcgan':
            return img, right_emb, wrong_emb
        elif self.method == 'acgan':
            return img, right_emb, right_h, right_e

    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _build_dict(self, tags):
        # construct a unordered set
        hair, eyes = set(), set()
        for t in tags:
            attr = re.search(r'\d+,(.*) hair (.*) eyes', t)
            a1, a2 = attr.group(1), attr.group(2)
            hair.add(a1), eyes.add(a2)
        # use the set to build dict(attr to index)
        dic_hair = {h: i for (i, h) in enumerate(sorted(hair))}
        dic_eyes = {e: i for (i, e) in enumerate(sorted(eyes))}
        # rgb_hair = [rgb(h) for h in sorted(hair)]
        # rgb_eyes = [rgb(e) for e in sorted(eyes)]

        return dic_hair, dic_eyes # , rgb_hair, rgb_eyes

    def _random_butnot(self, h_code, e_code):
        # randomly choose a pair of code except for (h_code, e_code)
        while True:
            rand_h, rand_e = random.randint(0, 11), random.randint(0, 9)
            if rand_h != h_code or rand_e != e_code:
                return rand_h, rand_e
    def __len__(self):
        return len(self.img_list)

class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, n_emb, method, batch_size, shuffle, validation_split, num_workers = 0, training=True):
        dataset = Dataset(data_dir, n_emb, method)
        collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)

