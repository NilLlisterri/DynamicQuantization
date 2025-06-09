import json
import os
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


class KWSDataset(Dataset):
    keywords = ["montserrat", "pedraforca", "vermell", "blau"]
    samples_folder = "./data/keywords"

    def __init__(self, transform=None):
        files = []
        for i, word in enumerate(self.keywords):
            file_list = os.listdir(f"{self.samples_folder}/{word}")
            random.shuffle(file_list)
            files.append(list(map(lambda f: f"{word}/{f}", file_list)))

        self.samples = list(sum(zip(*files), ()))
        self.transform = transform

        self.mean = -0.7344476580619812
        self.std = 2404.997314453125
        # all_features = []
        # for i in range(0, self.__len__()):
        #     data = self.__get_raw_values__(i)
        #     all_features.append(data) # Remove batch dim
        # all_features_tensor = torch.stack(all_features)
        #
        # self.mean = all_features_tensor.mean()
        # self.std = all_features_tensor.std()


    def __len__(self):
        return len(self.samples)

    def __get_raw_values__(self, idx):
        sample = self.samples[idx]
        with open(f'{self.samples_folder}/{sample}') as f:
            data = json.load(f)
            return Tensor(data['payload']['values'])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample.split('/')[0]
        target = self.keywords.index(label)

        with open(f'{self.samples_folder}/{sample}') as f:
            data = json.load(f)
            values = Tensor(data['payload']['values'])

        if self.transform:
            values = self.transform(values)

        standardized_values = (values - self.mean) / (self.std + 1e-8)  # Add epsilon to avoid div-by-zero

        return standardized_values, target
