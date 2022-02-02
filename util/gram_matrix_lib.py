from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import time
from util.metrics import compute_traditional_ood, compute_in
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model


LAYERS = 5

def forward_fun(args):
    def forward_all_feat(inputs, model):
        if args.model_arch.find('resnet') > -1:
            x = model.relu(model.bn1(model.conv1(inputs)))
            if args.in_dataset == 'imagenet':
                x = model.maxpool(x)
            feat_init = x.detach()
            x = model.layer1(x)
            feat_layer1 = x.detach()
            x = model.layer2(x)
            feat_layer2 = x.detach()
            x = model.layer3(x)
            feat_layer3 = x.detach()
            x = model.layer4(x)
            feat_layer4 = x.detach()
            x = model.avgpool(x)
            x = x.clip(max=args.threshold)
            feat_final = x.view(x.size(0), -1)

            return [feat_init, feat_layer1, feat_layer2, feat_layer3, feat_layer4, feat_final, model.fc(feat_final)]
            # {
            # "feat_init": feat_init,
            # "feat_layer1": feat_layer1,
            # "feat_layer2": feat_layer2,
            # "feat_layer3": feat_layer3,
            # "feat_layer4": feat_layer4,
            # "feat_final": feat_final,
            # "logits": model.fc(feat_final),
            # }
        elif args.model_arch == 'mobilenet':
            feat4 = model.features[:4](inputs)
            feat8 = model.features[4:8](feat4)
            feat12 = model.features[8:12](feat8)
            feat16 = model.features[12:16](feat12)
            feat_final = model.features[16:](feat16)

            # feat_final = model.features(inputs)
            x = model.avgpool(feat_final)
            x = x.reshape(x.shape[0], -1)
            # x = x.clip(max=threshold)
            logits = model.classifier(x)
            return [feat4, feat8, feat12, feat16, feat_final, logits]

    return forward_all_feat



def G_p(ob, p):
    temp = ob.detach()

    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp


args = get_args()
forward_feat = forward_fun(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)



def estimate_minmax(model, loaderIn, num_classes, powers=range(1, 10), cache=None):

    Mins = 1e10 * np.ones((num_classes, LAYERS, len(powers), 2048), dtype=np.float16)
    Maxs = -1e10 * np.ones((num_classes, LAYERS, len(powers), 2048), dtype=np.float16)

    count = 0
    for j, data in enumerate(loaderIn):

        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        inputs = images.float()

        with torch.no_grad():
            feat_list = forward_feat(inputs, model)

        for f_ind, feat in enumerate(feat_list[:LAYERS]):
            for p_ind, p in enumerate(powers):
                g_p = G_p(feat, p)

                current_min = g_p.min(dim=0)[0]
                current_max = g_p.max(dim=0)[0]

                for l_ind, label in enumerate(labels):
                    label = label.item()
                    Mins[label][f_ind][p_ind][:len(current_min)] = np.minimum(current_min.data.cpu().numpy(), Mins[label][f_ind][p_ind][:len(current_min)])
                    Maxs[label][f_ind][p_ind][:len(current_max)] = np.maximum(current_max.data.cpu().numpy(), Maxs[label][f_ind][p_ind][:len(current_max)])

        count += len(images)
        print(count)
        if count > 100000:
            break

    if cache is not None:
        np.save(cache, (Mins, Maxs))

    return Mins, Maxs

def estimate_deviations(model, loaderIn, Mins, Maxs, powers=range(1,10), cache=None):
    deviations = []
    count = 0

    Eva = np.zeros(LAYERS)
    SAMPLES = 10000

    for j, data in enumerate(loaderIn):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        inputs = images.float()

        feat_list = forward_feat(inputs, model)
        pred_labels = feat_list[-1].max(1)[1]

        for f_ind, feat in enumerate(feat_list[:LAYERS]):
            for p_ind, p in enumerate(powers):
                g_p = G_p(feat, p).data.cpu().numpy()
                dim = g_p.shape[1]
                for l_ind, label in enumerate(pred_labels):
                    label = label.item()
                    Eva[f_ind] += (np.maximum(Mins[label][f_ind][p_ind][:dim] - g_p[l_ind], 0) / np.abs(Mins[label][f_ind][p_ind][:dim] + 10 ** -6)).sum()
                    Eva[f_ind] += (np.maximum(g_p[l_ind] - Maxs[label][f_ind][p_ind][:dim], 0) / np.abs(Maxs[label][f_ind][p_ind][:dim] + 10 ** -6)).sum()

        count += len(images)
        print(count)
        if count > SAMPLES:
            break

    Eva = Eva / SAMPLES
    if cache is not None:
        np.save(cache, Eva)
    return Eva


def compute_deviations(model, inputs, Mins, Maxs, Eva, powers=range(1,10)):

    feat_list = forward_feat(inputs, model)
    pred_labels = feat_list[-1].max(1)[1]
    confs = torch.softmax(feat_list[-1], 1).max(1)[0]

    devs = np.zeros(len(inputs))
    for f_ind, feat in enumerate(feat_list[:LAYERS]):
        for p_ind, p in enumerate(powers):
            g_p = G_p(feat, p).data.cpu().numpy()
            dim = g_p.shape[1]
            for l_ind, label in enumerate(pred_labels):
                label = label.item()
                devs[l_ind] += (np.maximum(Mins[label][f_ind][p_ind][:dim] - g_p[l_ind], 0) / np.abs(
                    Mins[label][f_ind][p_ind][:dim] + 10 ** -6)).sum() / Eva[f_ind] / confs[l_ind]
                devs[l_ind] += (np.maximum(g_p[l_ind] - Maxs[label][f_ind][p_ind][:dim], 0) / np.abs(
                    Maxs[label][f_ind][p_ind][:dim] + 10 ** -6)).sum() / Eva[f_ind] / confs[l_ind]

    return devs


if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()
    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only

    loader_in_dict = get_loader_in(args, split=('train'))
    trainloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.num_classes
    model = get_model(args, num_classes, load_ckpt=True)


    os.makedirs("output/ood_scores/{}/{}".format(args.in_dataset, args.name), exist_ok=True)
    cache_name = "output/ood_scores/{}/{}/minmax.npy".format(args.in_dataset, args.name)
    if not os.path.exists(cache_name):
        Mins, Maxs = estimate_minmax(model, trainloaderIn, num_classes, cache=cache_name)
    else:
        Mins, Maxs = np.load(cache_name)

    loader_in_dict = get_loader_in(args, split=('val'))
    valloaderIn, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    cache_name = "output/ood_scores/{}/{}/eva.npy".format(args.in_dataset, args.name)
    if not os.path.exists(cache_name):
        Eva = estimate_deviations(model, valloaderIn, Mins, Maxs, cache=cache_name)
    else:
        Eva = np.load(cache_name)


