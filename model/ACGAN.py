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
        self.conv = nn.Sequential(
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
        self.real_dense = nn.Sequential(
            nn.Linear((nf * 8) * 4 * 4, nf * 8),
            nn.LeakyReLU(),
            nn.Linear(nf * 8, 1),
            nn.Sigmoid()
        )
        self.hair_dense = nn.Sequential(
            nn.Linear((nf * 8) * 4 * 4, nf * 8),
            nn.LeakyReLU(),
            nn.Linear(nf * 8, 12),
            nn.LogSoftmax()
        )
        self.eye_dense = nn.Sequential(
            nn.Linear((nf * 8) * 4 * 4, nf * 8),
            nn.LeakyReLU(),
            nn.Linear(nf * 8, 10),
            nn.LogSoftmax()
        )
        self._init()
    def forward(self, image):
        out = self.conv(image).reshape(image.shape[0], -1)
        real = self.real_dense(out)
        hair = self.hair_dense(out)
        eye = self.eye_dense(out)
        return real, hair, eye
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
    