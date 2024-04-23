import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

models_vers = [5] # for example: models_vers = [5] or models_vers = [3, 4, 5]
epsilon = 70
lambda_1 = 1
lambda_2 = 10
seed = 42
img_size = (640, 640)
batch_size = 8
num_workers = 4
BDD_IMG_DIR = 'bdd-dataset/data/train'
# BDD_IMG_DIR = 'TT100K/data/train'
# BDD_IMG_DIR = 'VOCdevkit/VOC2012/JPEGImages'

# 是否使用分割扰动块
use_patch = False
patch_size = (640, 640)


import torch
import random
import numpy

from datasets.augmentations1 import train_transform
from datasets.split_data_set_combined import SplitDatasetCombined_BDD

def collate_fn(batch):
    return tuple(zip(*batch))

def set_random_seed(seed_value, use_cuda=True):
    numpy.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

set_random_seed(seed)

split_dataset = SplitDatasetCombined_BDD(
            img_dir= BDD_IMG_DIR,
            img_size=img_size,
            transform=train_transform,
            collate_fn=collate_fn)

train_loader, val_loader, test_loader = split_dataset(val_split=0.1,
                                                      shuffle_dataset=True,
                                                      random_seed=seed,
                                                      batch_size=batch_size,
                                                      ordered=False)


from attack.sponge_attack import UniversalSpongePatch

torch.cuda.empty_cache()

patch_name = r"yolov"
for ver in models_vers:
  patch_name += f"_{ver}"
patch_name += f"_epsilon={epsilon}_lambda1={lambda_1}_lambda2={lambda_2}"
if use_patch:
    patch_name += f'_patch_{patch_size[0]}x{patch_size[1]}'
print(patch_name)

uap_phantom_sponge_attack = UniversalSpongePatch(patch_folder=patch_name, train_loader=train_loader, 
                      val_loader=val_loader, epsilon = epsilon,lambda_1=lambda_1,lambda_2=lambda_2,
                      img_size=img_size, patch_size=patch_size, models_vers=models_vers, use_patch=use_patch)
adv_img = uap_phantom_sponge_attack.run_attack()