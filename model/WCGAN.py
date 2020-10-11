import torch.nn as nn
import torch
from .base_model import BaseModel

class Generator(BaseModel):
    def __init__(self, noise_dim = 100, condition_dim = 120, nf = 64):
        super(Generator, self).__init__()
        self.text_dense = nn.Sequential(
            nn.Linear(condition_dim, nf * 4),
            nn.LeakyReLU()
        )
        self.cat_dense = nn.Linear(noise_dim + nf * 4, 4 * 4 * (nf * 8))
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features = nf * 8, momentum = 0.9),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = nf * 8, out_channels = nf * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = nf * 4, momentum = 0.9),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = nf * 4, out_channels = nf * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = nf * 2, momentum = 0.9),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = nf * 2, out_channels = nf, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = nf, momentum = 0.9),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels = nf, out_channels = 3, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
        self.nf = nf
        self._init()
    def forward(self, noise, condition):
        # print('noise', noise.shape)
        # print('condition', condition.shape)
        text_emb = self.text_dense(condition)
        # print('text_emb', text_emb.shape)
        cat = torch.cat((text_emb, noise), dim = 1)
        # print('cat', cat.shape)
        conv_in = self.cat_dense(cat).reshape(noise.shape[0], self.nf * 8, 4, 4)
        # print('conv_in', conv_in.shape)
        conv_out = self.conv(conv_in)
        # print('conv_out', conv_out.shape)
        return conv_out
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
            	m.weight.data.normal_(1.0, 0.02)
            	if m.bias is not None:
                    m.bias.data.zero_()

class Discriminator(BaseModel):
    def __init__(self, noise_dim = 100, condition_dim = 120, nf = 64):
        super(Discriminator, self).__init__()
        self.text_dense = nn.Sequential(
            nn.Linear(condition_dim, nf * 4), 
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = nf, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = nf, out_channels = nf * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = nf * 2, momentum = 0.9),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = nf * 2, out_channels = nf * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = nf * 4, momentum = 0.9),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = nf * 4, out_channels = nf * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = nf * 8, momentum = 0.9),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = nf * 8 + nf * 4, out_channels = nf * 8, kernel_size=1, stride=(1,1)),
            nn.BatchNorm2d(num_features = nf * 8, momentum = 0.9),
            nn.LeakyReLU(),
        )
        self.out_dense = nn.Sequential(
            nn.Linear((nf * 8) * 4 * 4, nf * 8),
            nn.LeakyReLU(),
            nn.Linear(nf * 8, 1),
            # nn.Sigmoid()
        )
        self.nf = nf
        self._init()
    def forward(self, image, condition):
        text_emb = self.text_dense(condition).reshape(condition.shape[0], 1, 1, self.nf * 4).repeat(1, 4, 4, 1)
        # print('text_emb', text_emb.shape)
        text_emb = text_emb.view(image.shape[0], self.nf * 4, 4, 4)
        # print('text_emb', text_emb.shape)
        conv_mid = self.conv1(image)
        # print('conv_mid', conv_mid.shape)
        cat = torch.cat((text_emb, conv_mid), dim = 1)
        # print('cat', cat.shape)
        conv_out = self.conv2(cat).reshape(image.shape[0], -1)
        # print('conv_out', conv_out.shape)
        out = self.out_dense(conv_out)
        # print('out', out.shape)
        return out
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
            	m.weight.data.normal_(1.0, 0.02)
            	if m.bias is not None:
                    m.bias.data.zero_()
    