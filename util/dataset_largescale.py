from torch.utils.data import Dataset
from PIL import Image
import io
import os
import random
import numpy as np
import pickle


def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img


class DatasetWithMeta(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(DatasetWithMeta, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()

        for line in lines:
            segs = line.strip().split(' ')
            self.images.append(' '.join(segs[:-1]))
            self.cls_idx.append(int(segs[-1]))
            self.classes.add(int(segs[-1]))
        self.num = len(self.images)
        # self.classes = len(self.cls_set)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, self.cls_idx[idx]


class DatasetWithMetaGroup(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, num_group=8):
        super(DatasetWithMetaGroup, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()
        self.num_group = num_group

        for line in lines:
            segs = line.strip().split(' ')
            self.images.append(' '.join(segs[:-2]))

            group_idx = int(segs[-2])
            sub_cls_idx = int(segs[-1])

            self.cls_idx.append((group_idx, sub_cls_idx))
            self.classes.add((group_idx, sub_cls_idx))

        self.num = len(self.images)
        # self.classes = len(self.cls_set)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)

        group_id, cls_id = self.cls_idx[idx]
        labels = np.zeros(self.num_group, dtype=np.int)
        labels[group_id] = cls_id + 1

        return img, labels


class DatasetWithMetaIndex(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(DatasetWithMetaIndex, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()

        for line in lines:
            segs = line.strip().split(' ')
            self.images.append(' '.join(segs[:-1]))
            self.cls_idx.append(int(segs[-1]))
            self.classes.add(int(segs[-1]))
        self.num = len(self.images)
        # self.classes = len(self.cls_set)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, self.cls_idx[idx], idx


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


class CIFAR100Group(Dataset):
    def __init__(self, data_file, meta_file, transform=None, num_group=20):
        super(CIFAR100Group, self).__init__()
        self.transform = transform

        self.data = unpickle(data_file)
        self.images = self.data[b'data']
        self.fine_labels = self.data[b'fine_labels']

        with open(meta_file) as f:
            lines = f.read().splitlines()

        # self.cls_idx = []
        # self.classes = set()
        self.num_group = num_group
        self.clsidx2groupidx = {}

        for line in lines[1:]:
            segs = line.strip().split('\t')
            old_cls_idx = int(segs[0])
            group_idx = int(segs[-2])
            sub_cls_idx = int(segs[-1])
            self.clsidx2groupidx[old_cls_idx] = (group_idx, sub_cls_idx)

        self.num = len(self.images)
        self.classes = self.clsidx2groupidx.items()
        # self.classes = len(self.cls_set)
        self.images = self.images.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # filename = os.path.join(self.root_dir, self.images[idx])

        # try:
        #     img = pil_loader(filename)
        # except:
        #     print(filename)
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))

        img = self.images[idx]
        img = Image.fromarray(img)
        # transform
        if self.transform is not None:
            img = self.transform(img)

        group_id, cls_id = self.clsidx2groupidx[self.fine_labels[idx]]
        labels = np.zeros(self.num_group, dtype=np.int)
        labels[group_id] = cls_id + 1

        return img, labels


class TinyImages(Dataset):

    def __init__(self, file_path, transform=None, exclude_cifar=True):

        data_file = open(file_path, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('data_lists/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 79302017