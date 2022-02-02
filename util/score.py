import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.mahalanobis_lib import get_Mahalanobis_score, get_ModedMaha_score
from util.gram_matrix_lib import compute_deviations
from util.influence_lib import calc_loss
from torch.autograd import grad


default_forward = lambda inputs, model: model(inputs)

def get_msp_score(inputs, model, forward_func, method_args, logits=None):
    if logits is None:
        with torch.no_grad():
            logits = forward_func(inputs, model)
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_energy_score(inputs, model, forward_func, method_args, logits=None):
    if logits is None:
        with torch.no_grad():
            logits = forward_func(inputs, model)

    # Using temperature scaling
    # outputs = outputs / temper
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores

def get_lgst_score(inputs, model, method_args, logits=None):
    with torch.no_grad():
        logits = model(inputs)
        scores = model.fc_lgst(logits).detach().cpu().numpy()[:, 0]
        # scores = F.softmax(ood_score, dim=1).detach().cpu().numpy()[:, 0]
    return scores

def get_exlgst_score(inputs, model, method_args):
    with torch.no_grad():
        feat = model.features(inputs)
        feat = model.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        scores = model.fc_lgst(feat).detach().cpu().numpy()[:, 0]
        # scores = F.softmax(ood_score, dim=1).detach().cpu().numpy()[:, 0]
    return scores


def get_infl_score(inputs, model, method_args):
    s_stat = method_args['s_stat']
    # logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)


    scores = [None for _ in range(len(inputs))]
    for i in range(len(inputs)):
        model.zero_grad()
        y = model(inputs[[i]].cuda())
        t = y.argmax().unsqueeze(0)
        loss = calc_loss(y, t)
        # Compute sum of gradients from model parameters to loss
        params = [p for p in model.parameters() if p.requires_grad]
        g_list = list(grad(loss, params, create_graph=True))
        scores[i] = -abs(sum([torch.sum(s_test * g).item() for s_test, g in zip(s_stat, g_list)]))
        # scores[i] = -abs(torch.sum(g_list[297] * s_stat[297]).item())
    return scores


def get_gradient_score(inputs, model, method_args):
    temper = method_args['temperature']
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)

    with torch.no_grad():
        feat = model.features(inputs)
        feat = model.avgpool(feat).squeeze()
    scores = [None for _ in range(len(feat))]
    for i in range(len(feat)):
        model.zero_grad()
        outputs = model.fc(feat[[i]])
        targets = torch.ones((1, outputs.shape[1])).cuda()
        outputs = outputs / temper
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()

        grad = model.fc.weight.grad.data
        scores[i] = torch.sum(torch.abs(grad * method_args['grad_mask'].to(grad.device))).cpu().numpy()
    return scores


def get_odin_score(inputs, model, forward_func, method_args):
    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # outputs = model(inputs)
    outputs = forward_func(inputs, model)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        outputs = forward_func(tempInputs, model)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores

def get_godin_score(inputs, model, forward_func, method_args):
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    # outputs = model(inputs)
    outputs, _, _ = forward_func(inputs, model)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        _, hx, _ = forward_func(tempInputs, model)
    # Calculating the confidence after adding perturbations
    nnOutputs = hx.data.cpu()
    nnOutputs = nnOutputs.numpy()
    # nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    # nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores

def get_gram_score(inputs, model, method_args):
    Mins = method_args['Mins']
    Maxs = method_args['Maxs']
    Eva = method_args['Eva']
    return -compute_deviations(model, inputs, Mins, Maxs, Eva)

def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']

    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores

def get_modedmaha_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    layer_indices = method_args['layer_indices']
    precision_class = method_args['precision_class']
    p = method_args['p']
    if len(layer_indices) > 1:
        Mahalanobis_scores = get_ModedMaha_score(inputs, model, num_classes, sample_mean, precision, precision_class, magnitude, layer_indices=layer_indices, p=0)
        return -regressor.predict_proba(Mahalanobis_scores)[:, 1]
    else:
        return get_ModedMaha_score(inputs, model, num_classes, sample_mean, precision, precision_class, magnitude, layer_indices=layer_indices, p=p)[:, 0]

def get_cos_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs, all_pred=True)
    scores = np.max(outputs[0].detach().cpu().numpy(), axis=1)
    return scores


def get_score(inputs, model, method, method_args, forward_func=default_forward, logits=None):
    if method == "msp":
        scores = get_msp_score(inputs, model, forward_func, method_args, logits)
    elif method == "odin":
        scores = get_odin_score(inputs, model, forward_func, method_args)
    elif method == "godin":
        scores = get_godin_score(inputs, model, forward_func, method_args)
    elif method == "energy":
        scores = get_energy_score(inputs, model, forward_func, method_args, logits)
    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    elif method == "modedmaha":
        scores = get_modedmaha_score(inputs, model, method_args)
    elif method == "gram":
        scores = get_gram_score(inputs, model, method_args)
    elif method == "gradient":
        scores = get_gradient_score(inputs, model, method_args)
    elif method == "infl":
        scores = get_infl_score(inputs, model, method_args)
    elif method == "lgst":
        scores = get_lgst_score(inputs, model, method_args)
    elif method == "exlgst":
        scores = get_exlgst_score(inputs, model, method_args)
    elif method == "cos":
        scores = get_cos_score(inputs, model, method_args)
    return scores