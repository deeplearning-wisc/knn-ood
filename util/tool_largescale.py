import argparse
import torchvision as tv
import torch
from util.dataset_largescale import DatasetWithMeta, DatasetWithMetaGroup, TinyImages
import os
import numpy as np
import models.resnetv2 as resnetv2_models
import models.resnet as resnet_models
import sklearn.metrics as sk
from models.wideresnet import WideResNet
from models.densenet import DenseNet3
import logging
import logging.config

FOLDER_DATASET = {'imagenet2012': 'ILSVRC-2012/val',
                  'lsun_crop': 'LSUN_C',
                  'lsun_resize': 'LSUN_resize',
                  'isun': 'iSUN',
                  'textures': 'dtd/images',
                  }

LIST_DATASET = {'imagenet2012_animal': 'ILSVRC-2012',
                'imagenet2012_not_animal': 'ILSVRC-2012',
                'imagenet2012_subset': 'ILSVRC-2012/org',
                'places365': 'places365',
                'deep_fashion': 'deep_fashion/img',
                'inat_plantae': 'iNat',
                'nih': 'nih',
                'objectnet': 'ObjectNet/objectnet-1.0_cropped',
                'ood_held_out': '',
                'sun': 'SUN397',
                'places365_test': 'places365/test_256',
                'sun100': 'SUN397',
                'sun50': 'SUN50',
                'places100': 'places365_standard/data_large',
                'places50': 'places50',
                }

GROUP_DATASET = {'imagenet2012_group': 'ILSVRC-2012'}

DEFAULT_SETTING = {
    'in_datadir' : '/home/sunyiyou/dataset/',
    'out_datadir' : '/home/sunyiyou/dataset/',
    'in_dataset' : 'imagenet2012_subset',
    'out_dataset' : 'places50',
    'in_data_list' : '/home/sunyiyou/dataset/ILSVRC-2012/imagenet2012_val_list.txt',
    'out_data_list' : '/home/sunyiyou/dataset/places50/places50_selected_list.txt',
    'model' : 'BiT-S-R101x1',
    'model_type' : 'bit_finetune',
    'model_path' : '/media/sunyiyou/ubuntu-hdd1/model_zoo/bit_finetune_BiT-S-R101x1/', #_imagenet_posw
    'logdir' : 'log/test',
    'name' : 'test',
    'batch' : 16
}

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_datadir", default=DEFAULT_SETTING['in_datadir'],
                        help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", default=DEFAULT_SETTING['out_datadir'],
                        help="Path to the out-of-distribution data folder.")
    parser.add_argument("--in_dataset", default=DEFAULT_SETTING['in_dataset'])
    parser.add_argument("--out_dataset", default=DEFAULT_SETTING['out_dataset'])

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    parser.add_argument("--logdir", default=DEFAULT_SETTING['logdir'],
                        help="Where to log training info (small).")
    parser.add_argument("--batch", type=int, default=DEFAULT_SETTING['batch'],
                        help="Batch size.")
    parser.add_argument("--name", default=DEFAULT_SETTING['name'],
                        help="Name of this run. Used for monitoring and checkpointing.")
    # parser.add_argument("--bit_pretrained_dir", default="bit_pretrained_models",
    #                     help="Where to search for pretrained BiT models.")

    parser.add_argument("--model_type", type=str, default=DEFAULT_SETTING['model_type'])
    parser.add_argument("--model", default=DEFAULT_SETTING['model'],
                        help="Which variant to use; BiT-M gives best results.")
    parser.add_argument("--model_path", default=DEFAULT_SETTING['model_path'], type=str)

    parser.add_argument("--in_data_list", default=DEFAULT_SETTING['in_data_list'], type=str)
    parser.add_argument("--out_data_list", default=DEFAULT_SETTING['out_data_list'], type=str)

    parser.add_argument("--save-softmax-scores", dest='save_softmax_scores', action='store_true')


    # args for densenet and wideresnet
    parser.add_argument('--layers', default=100, type=int,
                        help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int,
                        help='depth of resnet')
    parser.add_argument('--width', default=4, type=int,
                        help='width of resnet')
    parser.add_argument('--growth', default=12, type=int,
                        help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--reduce', default=0.5, type=float,
                        help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                        help='To not use bottleneck block')

    return parser


def setup_logger(args):
    """Creates and returns a fancy logger."""
    # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    # Why is setting up proper logging so !@?#! ugly?
    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "stderr": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "logfile": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": os.path.join(args.logdir, args.name, "train.log"),
                "mode": "a",
            }
        },
        "loggers": {
            "": {
                "handlers": ["stderr", "logfile"],
                "level": "DEBUG",
                "propagate": True
            },
        }
    })
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    logger.info(args)
    return logger


def mk_id_ood(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = 512, 480

    if args.in_dataset == "cifar100":
        val_tx = tv.transforms.Compose([tv.transforms.Resize((32, 32)),
                                        tv.transforms.ToTensor()])

    elif args.model_type == 'resnet':
        val_tx = tv.transforms.Compose([
            tv.transforms.Resize((crop, crop)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    else:
        val_tx = tv.transforms.Compose([
            tv.transforms.Resize((crop, crop)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    in_set = create_dataset(args.in_dataset, args.in_datadir, val_tx, args.in_data_list, args)
    out_set = create_dataset(args.out_dataset, args.out_datadir, val_tx, args.out_data_list, args)

    # if args.examples_per_class is not None:
    #   logger.info(f"Looking for {args.examples_per_class} images per class...")
    #   indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    #   train_set = torch.utils.data.Subset(train_set, indices=indices)

    logger.info(f"Using a in-distribution set with {len(in_set)} images.")
    logger.info(f"Using a out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def create_dataset(dataset_name, datadir, trans, data_list=None, args=None):
    if dataset_name == "TinyImage":
        return TinyImages('/nobackup-slow/dataset/80million/tiny_images.bin', transform=trans)
    elif dataset_name == "cifar10":
        return tv.datasets.CIFAR10(os.path.join(datadir, 'cifarpy'), transform=trans, train=False, download=True)
    elif dataset_name == "cifar100":
        return tv.datasets.CIFAR100(os.path.join(datadir, 'cifarpy'), transform=trans, train=False, download=True)
    elif dataset_name == "svhn":
        return tv.datasets.SVHN(os.path.join(datadir, 'svhn'), split='test', transform=trans, download=True)
    elif dataset_name in FOLDER_DATASET:
        return tv.datasets.ImageFolder(os.path.join(datadir, FOLDER_DATASET[dataset_name]), trans)
    elif dataset_name in LIST_DATASET:
        return DatasetWithMeta(os.path.join(datadir, LIST_DATASET[dataset_name]), data_list, trans)
    elif dataset_name in GROUP_DATASET:
        return DatasetWithMetaGroup(os.path.join(datadir, GROUP_DATASET[dataset_name]), data_list, trans, args.num_groups)

    # elif dataset_name == "imagenet2012":
    #     return tv.datasets.ImageFolder(os.path.join(datadir, "val"), trans)
    # elif dataset_name == "imagenet2012_animal":
    #     return DatasetWithMeta(datadir, data_list, trans)
    # elif dataset_name == "imagenet2012_not_animal":
    #     return DatasetWithMeta(datadir, data_list, trans)
    # elif dataset_name == "nih":
    #     return tv.datasets.ImageFolder(datadir, trans)

    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                         f"{dataset_name} dataset in the PyTorch codebase. "
                         f"In principle, it should be easy to add :)")


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # classes = np.unique(y_true)
    # if (pos_label is None and
    #         not (np.array_equal(classes, [0, 1]) or
    #                  np.array_equal(classes, [-1, 1]) or
    #                  np.array_equal(classes, [0]) or
    #                  np.array_equal(classes, [-1]) or
    #                  np.array_equal(classes, [1]))):
    #     raise ValueError("Data is not binary and pos_label is not specified")
    # elif pos_label is None:
    #     pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def load_state_dict_custom(model, state_dict):
    state_dict_new = {}
    for k, v in state_dict.items():
        state_dict_new[k[len("module."):]] = v
    model.load_state_dict(state_dict_new, strict=True)


def load_model(model_type, model, model_path, logger, num_classes, args):
    if model_type == "wideresnet" or model_type == "densenet":
        normalizer = tv.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        model_path = os.path.join(model_path, 'checkpoint_best.pth.tar')
        logger.info(f"Loading model from {model_path}")
        if model_type == "densenet":
            model = DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                              bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
        else:
            model = WideResNet(args.depth, num_classes, widen_factor=args.width,
                               dropRate=args.droprate, normalizer=normalizer)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'], strict=True)
        #load_state_dict_custom(model, state_dict['state_dict'])
    elif model_type == "resnetv2":
        model_path = os.path.join(model_path, model + '.npz')
        logger.info(f"Loading model from {model_path}")
        model = resnetv2_models.KNOWN_MODELS[model](head_size=num_classes)
        model.load_from(np.load(model_path))
    elif model_type == "bit_finetune":
        model_path = os.path.join(model_path, 'bit.pth.tar')
        logger.info(f"Loading model from {model_path}")
        model = resnetv2_models.KNOWN_MODELS[model](head_size=num_classes, k=args.inference_k)
        state_dict = torch.load(model_path)
        model.load_state_dict_custom(state_dict['model'])
    else:
        logger.info(f"Loading model from {model_path}")
        model = resnet_models.KNOWN_MODELS[model](num_classes=num_classes)
        state_dict = torch.load(model_path)
        model.load_state_dict_custom(state_dict['state_dict'])
    return model


def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr
