import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from ylib.dataloader.tinyimages_80mn_loader import TinyImages
from ylib.dataloader.imagenet_loader import ImageNet
from ylib.dataloader.svhn_loader import SVHN
from util.dataset_largescale import DatasetWithMeta

from ylib.dataloader.random_data import GaussianRandom, LowFreqRandom

# from util.broden_loader import BrodenDataset, broden_collate, dataloader

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
    #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_train = transforms.Compose([
    # transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomResizedCrop(size=imagesize, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize([x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                      [x / 255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'imagenet': 1000}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
        "eval": {
            'transform_train': transform_test,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_test_largescale,
        },
    })[config_type]

    train_loader, val_loader, lr_schedule, num_classes = None, None, [50, 75, 90], 0
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root='./datasets/data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root='./datasets/data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root='./datasets/data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root='./datasets/data', train=False, download=True, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "imagenet":
        root = args.imagenet_root
        # Data loading code
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)

    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes_dict[args.in_dataset],
    })

def get_loader_out(args, dataset=('tim', 'noise'), config_type='default', split=('train', 'val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch_size
        },
    })[config_type]
    train_ood_loader, val_ood_loader = None, None

    if 'train' in split:
        if dataset[0].lower() == 'imagenet':
            train_ood_loader = torch.utils.data.DataLoader(
                ImageNet(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif dataset[0].lower() == 'tim':
            train_ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size
        imagesize = 224 if args.in_dataset in {'imagenet'} else 32
        if val_dataset == 'SVHN':
            val_ood_loader = torch.utils.data.DataLoader(SVHN('datasets/ood_data/svhn/', split='test', transform=transform_test, download=False),
                                                       batch_size=batch_size, shuffle=False,
                                                        num_workers=2)
        elif val_dataset == 'dtd':
            transform = config.transform_test_largescale if args.in_dataset in {'imagenet'} else config.transform_test
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="datasets/ood_data/dtd/images", transform=transform),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'places365':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="datasets/ood_data/places365/", transform=transform_test),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'CIFAR-100':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./datasets/data', train=False, download=True, transform=transform_test),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'CIFAR-10':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./datasets/data', train=False, download=True, transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=2)
            
        elif val_dataset == 'places50':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./datasets/ood_data/Places",
                                                 transform=config.transform_test_largescale), batch_size=batch_size,
                shuffle=False, num_workers=2)
        elif val_dataset == 'sun50':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./datasets/ood_data/SUN",
                                                 transform=config.transform_test_largescale), batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        elif val_dataset == 'inat':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./datasets/ood_data/iNaturalist",
                                                 transform=config.transform_test_largescale), batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        elif val_dataset == 'tim':
            val_ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=transform_test),
                batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'imagenet':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join('dataset/imagenet', 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'noise':
            val_ood_loader = torch.utils.data.DataLoader(
                GaussianRandom(image_size=imagesize, data_size=10000),
                batch_size=batch_size, shuffle=False, num_workers=2)
            # val_ood_loader = torch.utils.data.DataLoader(
            #     GaussianRandom(image_size=imagesize, data_size=10000, transform=config.transform_test_largescale),
            #     batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'lfnoise':
            val_ood_loader = torch.utils.data.DataLoader(
                LowFreqRandom(image_size=imagesize, data_size=10000),
                batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./datasets/ood_data/{}".format(val_dataset),
                                                          transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=2)

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
    })
