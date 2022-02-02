
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
from scipy import misc

class OELoss(nn.Module):
    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, x, y):
        return -(x.mean(1) - torch.logsumexp(x, dim=1))

class LinfPGDAttack:

    def __init__(
            self, model, eps=8.0, nb_iter=40,
            eps_iter=1.0, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, loss_func='CE', num_classes=10,
            elementwise_best=False):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.elementwise_best = elementwise_best
        self.model = model
        self.num_classes = num_classes

        if loss_func == 'CE':
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        elif loss_func == 'OE':
            self.loss_func = OELoss()
        else:
            assert False, 'Not supported loss function {}'.format(loss_func)

        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        self.model.eval()

        x = x.detach().clone()
        if y is not None:
            y = y.detach().clone()
            y = y.cuda()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        delta.requires_grad_()

        if self.elementwise_best:
            outputs = self.model(x)
            loss = self.loss_func(outputs, y)
            worst_loss = loss.data.clone()
            worst_perb = delta.data.clone()

        if self.rand_init:
            delta.data.uniform_(-self.eps, self.eps)
            delta.data = torch.round(delta.data)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

        for ii in range(self.nb_iter):
            adv_x = x + delta / 255.0
            outputs = self.model(adv_x)

            if self.targeted:
                target = ((y + torch.randint(1, self.num_classes, y.shape).cuda()) % self.num_classes).long()
                loss = -self.loss_func(outputs, target)
            else:
                loss = self.loss_func(outputs, y)

            if self.elementwise_best:
                cond = loss.data > worst_loss
                worst_loss[cond] = loss.data[cond]
                worst_perb[cond] = delta.data[cond]

            loss.mean().backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + grad_sign * self.eps_iter
            delta.data = torch.clamp(delta.data, min=-self.eps, max=self.eps)
            delta.data = (torch.clamp(x.data + delta.data / 255.0, min=self.clip_min, max=self.clip_max) - x.data) * 255.0

            delta.grad.data.zero_()

        if self.elementwise_best:
            adv_x = x + delta / 255.0
            outputs = self.model(adv_x)

            if self.targeted:
                target = ((y + torch.randint(1, self.num_classes, y.shape).cuda()) % self.num_classes).long()
                loss = -self.loss_func(outputs, target)
            else:
                loss = self.loss_func(outputs, y)

            cond = loss.data > worst_loss
            worst_loss[cond] = loss.data[cond]
            worst_perb[cond] = delta.data[cond]

            adv_x = x + worst_perb / 255.0
        else:
            adv_x = x + delta.data / 255.0

        return adv_x
