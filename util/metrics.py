from __future__ import print_function
import argparse
import os

import sys

from scipy import misc
import numpy as np
from ylib.yplot import plot_distrib

def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95

def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: '+ out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

    # print(' {val:6.2f}'.format(val=100.*results['FPR']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    # print(' {val:6.2f}'.format(val=100.*results['AUOUT']), end='')

def print_all_results_tab(results, datasets, method):
    [print('{:6.2f}\t{:6.2f}\t{:6.2f}\t'.format(100.*result['FPR'], 100.*result['AUROC'], 100.*result['AUIN']), end='') for result in results]


def print_all_results(results, datasets, method):
    mtypes = ['FPR', 'AUROC', 'AUIN']#['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = compute_average_results(results)
    print(' OOD detection method: ' + method)
    print('             ', end='')
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    for result, dataset in zip(results,datasets):
        print('\n{dataset:12s}'.format(dataset=dataset), end='')
        print(' {val:6.2f}'.format(val=100.*result['FPR']), end='')
        # print(' {val:6.2f}'.format(val=100.*result['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUIN']), end='')
        # print(' {val:6.2f}'.format(val=100.*result['AUOUT']), end='')

    print('\nAVG         ', end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['FPR']), end='')
    # print(' {val:6.2f}'.format(val=100.*avg_results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*avg_results['AUIN']), end='')
    print()
    # [print('{:6.2f}\t{:6.2f}\t{:6.2f}\t'.format(100.*result['FPR'], 100.*result['AUROC'], 100.*result['AUIN']), end='') for result in results]
    # print()
    # print(' {val:6.2f}\n'.format(val=100.*avg_results['AUOUT']), end='')

def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results

def compute_traditional_ood(base_dir, in_dataset, out_datasets, method, name, args=None):
    # print('Natural OOD')
    # print('nat_in vs. nat_out')

    known = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name), delimiter='\n')

    known_sorted = np.sort(known)
    num_k = known.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    all_results = []

    total = 0.0

    for out_dataset in out_datasets:
        novel = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset), delimiter='\n')

        in_cond = (novel>threshold).astype(np.float32)
        total += novel.shape[0]

        results = cal_metric(known, novel, method)
        plot_distrib(known, novel, path='{base_dir}/{in_dataset}/{method}/{name}/nat/{out_dataset}/distrib.png'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset),
                      title="{name} {in_dataset}/{out_dataset} {method}".format(in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset))
        all_results.append(results)

    # avg_results = compute_average_results(all_results)
    # print_results(avg_results, in_dataset, "All", name, method)
    print_all_results(all_results, out_datasets, method)
    return

def compute_stat(base_dir, in_dataset, out_datasets, method, name):
    # print('Natural OOD')
    # print('nat_in vs. nat_out')

    known = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name), delimiter='\n')

    print(f"ID mean: {known.mean()} std: {known.std()}")

    all_mean = []
    all_std = []
    for out_dataset in out_datasets:
        novel = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset), delimiter='\n')
        all_mean.append(novel.mean())
        all_std.append(novel.std())

    print(f"OOD mean: {sum(all_mean) / len(out_datasets)} std: {sum(all_std) / len(out_datasets)}")
    return

def compute_in(base_dir, in_dataset, method, name):

    known_nat = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name), delimiter='\n')
    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    known_nat_label = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_labels.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name))

    nat_in_cond = (known_nat>threshold).astype(np.float32)
    nat_correct = (known_nat_label[:,0] == known_nat_label[:,1]).astype(np.float32)
    nat_conf = np.mean(known_nat_label[:,2])
    known_nat_cond_acc = np.sum(nat_correct * nat_in_cond) / max(np.sum(nat_in_cond), 1)
    known_nat_acc = np.mean(nat_correct)
    known_nat_cond_fnr = np.sum(nat_correct * (1.0 - nat_in_cond)) / max(np.sum(nat_correct),1)
    known_nat_fnr = np.mean((1.0 - nat_in_cond))
    known_nat_eteacc = np.mean(nat_correct * nat_in_cond)

    # print('In-distribution performance:')
    print('FNR: {fnr:6.2f}, Acc: {acc:6.2f}, End-to-end Acc: {eteacc:6.2f}'.format(fnr=known_nat_fnr*100,acc=known_nat_acc*100,eteacc=known_nat_eteacc*100))
    # print('\t{acc:6.2f}, {eteacc:6.2f}'.format(fnr=known_nat_fnr*100,acc=known_nat_acc*100,eteacc=known_nat_eteacc*100))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--name', default="densenet", type=str,
                        help='neural network name and training set')
    parser.add_argument('--method', default='energy', type=str, help='odin mahalanobis')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--epsilon', default=8, type=int, help='epsilon')

    parser.set_defaults(argument=True)

    args = parser.parse_args()

    np.random.seed(1)

    out_datasets = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
    compute_traditional_ood(args.base_dir, args.in_dataset, out_datasets, args.method, args.name)
    compute_in(args.base_dir, args.in_dataset, args.method, args.name)