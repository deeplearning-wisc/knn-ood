import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="imagenet", type=str, help='CIFAR-10 imagenet')
    parser.add_argument('--out-datasets', default=['inat', 'sun50', 'places50', 'dtd', ], nargs="*", type=str, help="['SVHN', 'LSUN', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument('--name', default="resnet18-supcon", type=str, help='neural network name and training set')
    parser.add_argument('--model-arch', default='resnet18-supcon', type=str, help='model architecture')
    parser.add_argument('--p', default=0, type=float, help='sparsity level')
    parser.add_argument('--imagenet-root', default='./datasets/imagenet/', type=str, help='imagenet root')
    parser.add_argument('--seed', default=0, type=int, help='seed')

    parser.add_argument('--method', default='', type=str, help='')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save-epoch', default=10, type=int,
                        help='save the model every save_epoch')
    parser.add_argument('--cal-metric', help='calculatse metric directly', action='store_true')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
    parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int, help='depth of resnet')
    parser.add_argument('--width', default=4, type=int, help='width of resnet')
    parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
    parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args