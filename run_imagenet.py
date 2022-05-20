import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np

args = get_args()

seed = args.seed
print(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class_num = 1000
id_train_size = 1281167
id_val_size = 50000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 2048))
score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))
label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))


cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 2048))
score_log_val = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))
label_log_val = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
FORCE_RUN = False
norm_cache = f"cache/{args.in_dataset}_train_{args.name}_in/feat_norm.mmap"
if not FORCE_RUN and os.path.exists(norm_cache):
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_train_size, 2048))
else:
    feat_log_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_train_size, 2048))
    feat_log_norm[:] = normalizer(feat_log)

norm_cache = f"cache/{args.in_dataset}_val_{args.name}_in/feat_norm.mmap"
if not FORCE_RUN and os.path.exists(norm_cache):
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_val_size, 2048))
else:
    feat_log_val_norm = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_val_size, 2048))
    feat_log_val_norm[:] = normalizer(feat_log_val)


prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))
ftrain = np.ascontiguousarray(feat_log_norm.astype(np.float32))
ftest = np.ascontiguousarray(feat_log_val_norm.astype(np.float32))


ood_feat_log_all = {}
food_all = {}
sood_all = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}

for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 2048))
    ood_score_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], class_num))
    ood_feat_log_all[ood_dataset] = ood_feat_log
    food_all[ood_dataset] = prepos_feat(ood_feat_log).astype(np.float32)

#################### KNN score OOD detection #################

ALPHA = 1.00
for K in [1000]:
    rand_ind = np.random.choice(id_train_size, int(id_train_size * ALPHA), replace=False)
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain[rand_ind])

    ################### Using KNN distance Directly ###################
    if True:
        D, _ = index.search(ftest, K, )
        scores_in = -D[:,-1]
        all_results = []
        for ood_dataset, food in food_all.items():
            D, _ = index.search(food, K)
            scores_ood_test = -D[:,-1]
            results = metrics.cal_metric(scores_in, scores_ood_test)
            all_results.append(results)

        metrics.print_all_results(all_results, args.out_datasets, 'KNN')
        print()
