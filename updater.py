import chainer
import chainer.functions as F

import numpy as np

from utils import data_process

class Pix2pixUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.xp = self.G.xp
        self.lambd = kwargs.pop('lambd')
        self.eps = 10**-12
        super(Pix2pixUpdater, self).__init__(*args, **kwargs)

    def loss_D(self, real_D, fake_D):
        batch_size, _, h, w = real_D.shape

        loss = - F.sum(F.log(real_D + self.eps) + F.log(1 - fake_D + self.eps)) / (batch_size*h*w)
        chainer.report({'loss': loss}, self.D)

        return loss

    def loss_G(self, real_B, fake_B, fake_D):
        loss_l1 = F.mean_absolute_error(real_B, fake_B)
        chainer.report({'loss_l1': loss_l1}, self.G)

        batch_size, _, h, w = fake_D.shape
        loss_D = - F.sum(F.log(fake_D + self.eps)) / (batch_size*h*w)
        chainer.report({'loss_D': loss_D}, self.G)

        loss = loss_D + self.lambd*loss_l1
        chainer.report({'loss': loss}, self.G)

        return loss

    def update_core(self):
        batch = self.get_iterator('main').next()
        A = data_process([A for A,B in batch], self.converter, self.device)
        B = data_process([B for A,B in batch], self.converter, self.device)

        real_AB = F.concat((A, B))

        fake_B = self.G(A, test=False)

        fake_AB = F.concat((A, fake_B))

        real_D = self.D(real_AB, test=False)
        fake_D = self.D(fake_AB, test=False)
        
        optimizer_G = self.get_optimizer('main')
        optimizer_D = self.get_optimizer('D')
        optimizer_D.update(self.loss_D, real_D, fake_D)
        optimizer_G.update(self.loss_G, B, fake_B, fake_D)