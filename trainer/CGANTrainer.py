import numpy as np
import torch
import torch.nn.functional as F
import logging
import os
import random
import torch.nn as nn
import math
from torchvision.utils import save_image
from utils import embed

class CGANTrainer:    
    def __init__(self, noise_dim, n_emb, gen, dis, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.n_epochs = opt.n_epochs
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.noise_dim = noise_dim
        self.device = self._prepare_gpu()
        self.gen = gen
        self.dis = dis
        self.gen_iter = 1
        self.dis_iter = 1
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr = opt.glr, betas=(0.5, 0.999))
        self.dis_optimizer = torch.optim.Adam(self.dis.parameters(), lr = opt.dlr, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(120, self.noise_dim, device = self.device)
        
        self.fixed_condition = torch.zeros(120, n_emb, device = self.device)

        for i in range(12):
            for j in range(10):
                self.fixed_condition[i * 10 + j] = embenn.BCELoss()d(i, j, n_emb = n_emb).to(self.device)

        self.criterion = nn.BCELoss()
        self.real_label = 1
        self.fake_label = 0
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.gen.to(self.device)
        self.dis.to(self.device)
        self.logger.info('[GEN_STRUCTURE]')
        self.logger.info(self.gen)
        self.logger.info('[DIS_STRUCTURE]')
        self.logger.info(self.dis)

        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._train_epoch(i)
            merged_log = {**log}
            all_log.append(merged_log)
            checkpoint = {
                'log': all_log,
                'gen_state_dict': self.gen.state_dict(),
                'dis_state_dict': self.dis.state_dict(),
            }
            self.logger.info("======================================================================================")
            self.logger.info('FINISH EPOCH: [%d/%d] Loss_D: %.4f Loss_G: %.4f'% (i, self.begin_epoch + self.n_epochs, log['Dis_loss'], log['Gen_loss']))
            self.logger.info("======================================================================================")
            if i % 1 == 0:
                check_path = os.path.join(opt.save_dir, 'checkpoint_' + str(i) + '.pth')
                torch.save(checkpoint, check_path)
                with torch.no_grad():
                    fixed_image = self.gen(self.fixed_noise, self.fixed_condition)
                    save_image(fixed_image.data[:12*10], os.path.join(opt.save_dir, "images_%d.png" % i), nrow = 10, normalize = True)

    def _train_epoch(self, epoch):
        self.gen.train()
        self.dis.train()
        G_sum_loss = 0
        D_sum_loss = 0

        for batch_idx, (real_images, right_emb, wrong_emb)  in enumerate(self.dataloader):
            real_label = torch.ones(real_images.shape[0]).to(self.device)
            fake_label = torch.zeros(real_images.shape[0]).to(self.device)
            # ===================
            # Train Discriminator
            # ===================
            # for p in self.dis.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            
            self.dis_optimizer.zero_grad()
            real_images = real_images.to(self.device)
            right_emb = right_emb.to(self.device)
            wrong_emb = wrong_emb.to(self.device)

            input_noise = torch.randn(real_images.size()[0], self.noise_dim).to(self.device)
            fake_images = self.gen(input_noise, right_emb).detach()

            real_predict = self.dis(real_images, right_emb)
            fake_predict = self.dis(fake_images, right_emb)
            wrong_predict = self.dis(real_images, wrong_emb)

            real_loss = self.criterion(real_predict, real_label)
            fake_loss = self.criterion(fake_predict, fake_label)
            wrong_loss = self.criterion(wrong_predict, fake_label)

            D_x = real_predict.mean().item()
            D_G_z1 = fake_predict.mean().item()
            loss_d = real_loss + (wrong_loss + fake_loss) / 2

            loss_d.backward()
            self.dis_optimizer.step()
            # ==================
            # Train Generator
            # ==================

            self.gen_optimizer.zero_grad()
            input_noise = torch.randn(real_images.size()[0], self.noise_dim).to(self.device)
            fake_images = self.gen(input_noise, right_emb)
            fake_predict = self.dis(fake_images, right_emb)
            loss_g = self.criterion(fake_predict, real_label)
            D_G_z2 = fake_predict.mean().item()
            loss_g.backward()
            self.gen_optimizer.step()

            G_sum_loss += loss_g.item()
            D_sum_loss += loss_d.item()
            self.logger.info('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'%\
             (batch_idx + 1, len(self.dataloader), loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))
        
        log = {
            'epoch': epoch,
            'Gen_loss': G_sum_loss,
            'Dis_loss': D_sum_loss
        }

        return log

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.dis.load_state_dict(checkpoint['dis_state_dict'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')