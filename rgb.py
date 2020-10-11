import logging
from model import CGAN, ACGAN
import numpy as np
import math
import torch
import argparse
import utils
import os
from torchvision.utils import save_image
from utils import embed, ensure_dir, rgb_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='saved/case5/checkpoint_21.pth', help='the path of checkpoint to be loaded')
parser.add_argument('--save_dir', type=str, default='./', help='the path to save result')
parser.add_argument('--method', type=str, default='cgan')
parser.add_argument('--nemb', type=int, default=6)
parser.add_argument('--nf', type=int, default=64)
opt = parser.parse_args()
ensure_dir(opt.save_dir)
NOISE_DIM = 100
NF = opt.nf
N_EMB = opt.nemb

if __name__ == '__main__':
    model = None
    if opt.method == 'cgan':
        model = CGAN
    elif opt.method == 'acgan':
        model = ACGAN
    
    G = model.Generator(noise_dim = NOISE_DIM, condition_dim = N_EMB, nf = NF)
    D = model.Discriminator(noise_dim = NOISE_DIM, condition_dim = N_EMB, nf = NF)
    g_state_dict = torch.load(opt.checkpoint)['gen_state_dict']
    d_state_dict = torch.load(opt.checkpoint)['dis_state_dict']
    G.load_state_dict(g_state_dict)
    D.load_state_dict(d_state_dict)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    
    G.to(device), G.eval()
    D.to(device), D.eval()
    noise_dim, n_emb = 100, N_EMB
    image_container = torch.tensor([], dtype = torch.float, device = device)
    fixed_noise = torch.randn(noise_dim, device = device).repeat(1, 1)   

    # y(x(a,b,c) + (1-x)(255,255,255))+(1-y)(0,0,0)
    # x(a,b,c) + (1-x)(255,255,255)
    # hr, hg, hb = 255, 122, 122
    er, eg, eb = 0, 0, 255

    red = np.array([255, 0, 0, er, eg, eb])
    black = np.array([0, 0, 0, er, eg, eb])
    white = np.array([255, 255, 255, er, eg, eb])
    
    stage = np.array([[0, 0, +1], [-1, 0 ,0], [0, +1, 0], [0, 0, -1], [+1, 0, 0], [0, -1, 0]])
    n_row = 15

    for a in range(2):
        mixed, l, r, d = None, None, None, None
        if a == 0:
            mixed = white
            l, r, d = 1, -0.1, -0.1
        else:
            mixed = black
            l, r, d = 0, 1 + 0.1, 0.1

        for x in np.arange(l, r, d):
            curr = (1 - x) * red + x * mixed
            # print(curr) 
            rgb_range = np.ceil(np.max(curr[0:3]) - np.min(curr[0:3]))
            sample_rate = rgb_range * len(stage) // n_row
            # print(curr)
            # print(rgb_range * len(stage))
            # input()
            # print(sample_rate)
            count = 0
            for s in range(len(stage)):
                for i in range(int(rgb_range)):
                    count += 1
                    condition = torch.from_numpy(curr).type(torch.float).to(device).repeat(1, 1) / 255
                    # generate
                    if count % sample_rate == 0:
                        with torch.no_grad():
                            # print(curr)
                            image = G(fixed_noise, condition)
                            image_container = torch.cat([image_container, image], dim = 0)
                    # update
                    curr[0:3] = curr[0:3] + stage[s]
                    # print(condition)
                    # input()
            # print(count)
            # print(image_container.shape[0])

    save_image(image_container, os.path.join(opt.save_dir, 'result.png'), nrow = n_row, normalize = True)