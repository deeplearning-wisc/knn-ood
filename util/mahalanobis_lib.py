from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        print(total)
        if total > 50000:
            break
        # data = data.cuda()
        data = Variable(data)
        data = data.cuda()
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).double().cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    precision_class_alllayers = []
    for k in range(num_output):
        precision_class = [None for _ in range(num_classes)]
        X = 0
        for cls in range(num_classes):
            X_cls = list_features[k][cls] - sample_class_mean[k][cls]
            if cls == 0:
                X = X_cls
            else:
                X = torch.cat((X, X_cls), 0)
            group_lasso.fit(X_cls.cpu().numpy())
            precision_class[cls] = torch.from_numpy(group_lasso.precision_).double().cuda()
        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).double().cuda()
        precision.append(temp_precision)
        precision_class_alllayers.append(precision_class)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision, precision_class_alllayers

def get_ModedMaha_score(inputs, model, num_classes, sample_mean, precision, precision_class, magnitude, layer_indices=[-1], p=0):

    if p > 0:
        assert len(layer_indices) == 1
        contrib = sample_mean[-1] * model.fc.weight.data
        _, class_topk_inds = contrib.topk(int((1-p/100) * sample_mean[-1].shape[1]), dim=1)

    Mahalanobis_scores = None
    layer_indices = list(range(layer_indices)) if type(layer_indices) == int else layer_indices
    for layer_index in layer_indices:
        data = Variable(inputs, requires_grad=True)
        data = data.cuda()

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for cls in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][cls]
            zero_f = out_features.data - batch_sample_mean
            inv_covmat = precision_class[layer_index][cls]
            # term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if p > 0:
                inds = class_topk_inds[cls]
                zero_f = zero_f[:, inds]
                inv_covmat = inv_covmat[:, inds][inds, :]
            term_gau = -0.5*torch.mm(torch.mm(zero_f, inv_covmat), zero_f.t()).diag()

            if cls == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # # Input_processing
        # sample_pred = gaussian_score.max(1)[1]
        # batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        # zero_f = out_features - Variable(batch_sample_mean)
        # pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision_class[layer_index][cls])), zero_f.t()).diag()
        # loss = torch.mean(-pure_gau)
        # loss.backward()
        #
        # gradient =  torch.ge(data.grad.data, 0)
        # gradient = (gradient.float() - 0.5) * 2
        #
        # tempInputs = torch.add(data.data, -magnitude, gradient)
        #
        # noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        # noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        # noise_out_features = torch.mean(noise_out_features, 2)
        # gaussian_score = 0
        # for i in range(num_classes):
        #     batch_sample_mean = sample_mean[layer_index][i]
        #     zero_f = noise_out_features.data - batch_sample_mean
        #     term_gau = -0.5*torch.mm(torch.mm(zero_f, precision_class[layer_index][cls]), zero_f.t()).diag()
        #     if i == 0:
        #         gaussian_score = term_gau.view(-1,1)
        #     else:
        #         gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        gaussian_score, _ = torch.max(gaussian_score, dim=1)
        gaussian_score = np.asarray(gaussian_score.cpu().numpy(), dtype=np.float32)

        if Mahalanobis_scores is None:
            Mahalanobis_scores = gaussian_score.reshape((gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate((Mahalanobis_scores, gaussian_score.reshape((gaussian_score.shape[0], -1))), axis=1)

    return Mahalanobis_scores


def get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude):

    for layer_index in range(num_output):
        data = Variable(inputs, requires_grad = True)
        data = data.cuda()

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, -magnitude, gradient)

        noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate((Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))), axis=1)

    return Mahalanobis_scores
