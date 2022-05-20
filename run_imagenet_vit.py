
from util import metrics
import faiss, time
from pytorch_pretrained_vit import ViT
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import torchvision
import os
from util.args_loader import get_args

# Data loading code
kwargs = {'num_workers': 2, 'pin_memory': True}
args = get_args()
root = args.imagenet_root

model = ViT('B_16_imagenet1k', pretrained=True)

def feat_extract():
    transformer = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    batch_size = 8
    trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transformer)
    rand_ind = np.random.choice(len(trainset), int(len(trainset) * 1.0), replace=False)

    trainset_subset = torch.utils.data.Subset(trainset, rand_ind)

    trainloaderIn = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size, shuffle=False, **kwargs)

    testloaderIn = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(root, 'val'), transformer),
        batch_size=batch_size, shuffle=False, **kwargs)

    FORCE_RUN = False
    featdim = 768
    num_classes = 1000
    device = 'cuda'
    model.to(device)

    for split, in_loader in [('val', testloaderIn), ('train', trainloaderIn)]:
        cache_dir = f"cache/imagenet_{split}_vit_in"

        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), featdim))
            score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset),))

            model.eval()
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)}")
                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                out = model.features(inputs)
                score = model.fc(out)
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                score_log[start_ind:end_ind] = score.data.cpu().numpy()
        else:
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), featdim))
            score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset),))


    for ood_dataset in args.out_datasets:
        from util.dataset_largescale import DatasetWithMeta

        if ood_dataset == 'dtd':
            out_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root="datasets/ood_data/dtd/images", transform=transformer),
                batch_size=batch_size, shuffle=True, num_workers=2)
        elif ood_dataset == 'places50':
            out_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./datasets/ood_data/Places",
                                                 transform=transformer),
                batch_size=batch_size, shuffle=True, num_workers=2)
        elif ood_dataset == 'sun50':
            out_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./datasets/ood_data/SUN",
                                                 transform=transformer),
                batch_size=batch_size, shuffle=True, num_workers=2)
        elif ood_dataset == 'inat':
            out_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("./datasets/ood_data/iNaturalist",
                                                 transform=transformer),
                batch_size=batch_size, shuffle=True, num_workers=2)

        cache_dir = f"cache/{ood_dataset}vsimagenet_vit_out"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), featdim))
            ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), num_classes))

            model.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)}")
                inputs = inputs.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                out = model.features(inputs)
                score = model.fc(out)
                # score = net(inputs)
                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
            # np.save(cache_name, (ood_feat_log.T, ood_score_log.T))
        else:
            # ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            # ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), featdim))
            ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), num_classes))


feat_extract()

id_train_size = 1281167
id_val_size = 50000
class_num = 1000

cache_dir = f"cache/{args.in_dataset}_train_vit_in"
feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 768))
score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))
label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))

cache_dir = f"cache/{args.in_dataset}_val_vit_in"
feat_log_val = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 768))
score_log_val = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))
label_log_val = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))


ood_feat_log_all = {}
food_all = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_vit_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 768))
    ood_score_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_vit_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], class_num))
    ood_feat_log_all[ood_dataset] = ood_feat_log
    food_all[ood_dataset] = prepos_feat(ood_feat_log).astype(np.float32)

ftest = prepos_feat(feat_log_val).astype(np.float32)
print("normalizing training log")
ftrain = prepos_feat(feat_log).astype(np.float32)


for K in [1000]:

    # rand_ind = np.random.choice(50000, 500, replace=False)
    index = faiss.IndexFlatL2(ftrain.shape[1])
    begin = time.time()
    index.add(ftrain)
    index_bad = index
    ################### Using KNN distance Directly ###################

    if True:
        # K = 3000
        D, _ = index_bad.search(ftest, K, )
        scores_in = -D[:, -1]
        all_results = []
        all_score_ood = []
        for ood_dataset, food in food_all.items():
            D, _ = index_bad.search(food, K)
            scores_ood_test = -D[:, -1]
            all_score_ood.extend(scores_ood_test)
            results = metrics.cal_metric(scores_in, scores_ood_test)
            all_results.append(results)

        print(time.time() - begin, end="\t\t")
        metrics.print_all_results(all_results, args.out_datasets, 'KNN')
        print()


# ################### SSD Method ###################
#
# if True:
#     begin = time.time()
#     inv_sigma_cls = [None for _ in range(class_num)]
#     covs_cls = [None for _ in range(class_num)]
#     mean_cls = [None for _ in range(class_num)]
#     cov = lambda x: np.cov(x.T, bias=True)
#     for cls in range(class_num):
#         mean_cls[cls] = ftrain[label_log == cls].mean(0)
#         feat_cls_center = ftrain[label_log == cls] - mean_cls[cls]
#         inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
#
#     if True:
#         def maha_score(X):
#             score_cls = np.zeros((class_num, len(X)))
#             for cls in range(class_num):
#                 inv_sigma = inv_sigma_cls[cls]
#                 z = X - mean_cls[cls]
#                 score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
#             return score_cls.max(0)
#
#
#         dtest = maha_score(ftest)
#         all_results = []
#         for name, food in food_all.items():
#             dood = maha_score(food)
#             results = metrics.cal_metric(dtest, dood)
#             all_results.append(results)
#
#         metrics.print_all_results(all_results, args.out_datasets, 'ssd')
#     print(time.time() - begin)
#
