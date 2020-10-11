from loader import DataLoader
from model import CGAN, ACGAN, WCGAN
from trainer import CGANTrainer, ACGANTrainer, WCGANTrainer
from utils import utils
import torch
import argparse
import logging
import os



parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='saved/')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--nemb', type=int, default=22)
parser.add_argument('--method', type=str, default='acgan')
parser.add_argument('--glr', type=float, default=0.0002)
parser.add_argument('--dlr', type=float, default=0.0002)
parser.add_argument('--nf', type=int, default=64)
parser.add_argument('--checkpoint', type=str, default=None)


opt = parser.parse_args()
utils.ensure_dir(opt.save_dir)

handlers = [logging.FileHandler(os.path.join(opt.save_dir, 'output.log'), mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
logger = logging.getLogger()

NOISE_DIM = 100
NF = opt.nf
N_EMB = opt.nemb

if __name__ == '__main__':
    L = DataLoader(data_dir = 'data/', n_emb = N_EMB, method = opt.method, batch_size = opt.batch, shuffle = True, validation_split = 0.0)
    model, trainer = None, None
    if opt.method == 'cgan':
        model = CGAN
        trainer = CGANTrainer
    elif opt.method == 'acgan':
        model = ACGAN
        trainer = ACGANTrainer
    elif opt.method == 'wcgan':
        model = WCGAN
        trainer = WCGANTrainer

    G = model.Generator(noise_dim = NOISE_DIM, condition_dim = N_EMB, nf = NF)
    D = model.Discriminator(noise_dim = NOISE_DIM, condition_dim = N_EMB, nf = NF)
    T = trainer(noise_dim = NOISE_DIM, n_emb = N_EMB, gen = G , dis = D, dataloader = L, opt = opt)
    T.train()