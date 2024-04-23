import cv2
import fnmatch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset
import random


class SplitDatasetCombined_BDD:
    def __init__(self, img_dir, img_size, transform=None, collate_fn=None):
        self.dataset_train = CustomDataset(type_ds='train',
                                     img_dir=img_dir,
                                     transform=transform)
        self.dataset_val = CustomDataset(type_ds='val',
                                           img_dir=img_dir,
                                           transform=transform)
        self.dataset_test = CustomDataset(type_ds='test',
                                           img_dir=img_dir,
                                           transform=transform)
        self.img_dir = img_dir

        self.collate_fn = collate_fn

    def __call__(self, val_split, shuffle_dataset, random_seed, batch_size, *args, **kwargs):

        train_indices, val_indices, test_indices = self.create_random_indices(val_split)

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(self.dataset_train, batch_size=batch_size,  sampler=train_sampler, collate_fn=self.collate_fn)
        validation_loader = DataLoader(self.dataset_val, batch_size=batch_size, sampler=valid_sampler, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.dataset_test,  sampler=test_sampler, collate_fn=self.collate_fn)

        return train_loader, validation_loader, test_loader

    def create_random_indices(self,val_split):

        all_indices = [i for i in range(5000)]
        total = 2000
        data_set_indices = random.sample(all_indices, k=total)
        train_val = 1500
        split_index = int(train_val * (1-val_split))
        train_indices = data_set_indices[0:split_index]
        val_indices = data_set_indices[split_index:train_val]
        test_indices = data_set_indices[train_val:total]

        return train_indices, val_indices, test_indices


class CustomDataset(Dataset):
    def __init__(self, type_ds, img_dir, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.shuffle = shuffle
        self.type_ds = type_ds
        self.img_names = self.get_image_names()
        self.transform = transform


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        
        image = transformed['image'].float()

        return (image, self.img_names[idx])

    def get_image_names(self):
        png_images = fnmatch.filter(os.listdir(self.img_dir), '*.png')
        jpg_images = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')

        return png_images + jpg_images




