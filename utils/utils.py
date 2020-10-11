import numpy as np
import torch
import random
import os


def embed(h_code, e_code, n_emb = 120):
    """
    Return the embeded tensor based on the code passed in
    """
    rgb_hair = [[0, 255, 255], [0, 0, 0], [226, 224, 110], [0, 0, 255], [160, 82, 45], [192, 192, 192], [0, 128, 0], [255, 165, 0], [255, 192, 203], [128, 0, 128], [255, 0, 0], [255, 255, 255]]
    rgb_eyes = [[0, 255, 255], [0, 0, 0], [0, 0, 255], [160, 82, 45], [0, 128, 0], [255, 165, 0], [255, 192, 203], [128, 0, 128], [255, 0, 0], [255, 255, 0]]
    if n_emb == 120:
        tensor = torch.zeros(12 * 10, dtype = torch.float)
        tensor[h_code * 10 + e_code] = 1
        return tensor
    elif n_emb == 22:
        # hhhhhhhhhhhheeeeeeeeee
        tensor = torch.zeros(22, dtype = torch.float)
        tensor[h_code] = 1
        tensor[12 + e_code] = 1
        return tensor
    elif n_emb == 6: 
        tensor_h = torch.tensor(rgb_hair[h_code], dtype = torch.float)
        tensor_e = torch.tensor(rgb_eyes[e_code], dtype = torch.float)
        tensor = torch.cat((tensor_h, tensor_e), dim = 0)
        tensor = tensor / 255 # normalize
        return tensor

def random_neglect(lower, upper, neglect):
    r = list(range(lower, neglect)) + list(range(neglect + 1, upper))
    x = random.choice(r)
    return x

def rgb(color_name):
    color_map = {'aqua': [0, 255, 255], 'black': [0, 0, 0], 'blonde': [226, 224, 110], 'blue': [0, 0, 255], 'brown': [160, 82, 45], 'gray': [192, 192, 192], 'green': [0, 128, 0], 'orange': [255, 165, 0], 'pink': [255, 192, 203], 'purple': [128, 0, 128], 'red': [255, 0 ,0], 'white': [255, 255, 255], 'yellow': [255, 255, 0]}
    return color_map[color_name]

def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)

def rgb_tensor(hr, hg, hb, er, eg, eb):
    tensor = torch.tensor([hr, hg, hb, er, eg, eb], dtype = torch.float)
    tensor = tensor / 255
    return tensor
