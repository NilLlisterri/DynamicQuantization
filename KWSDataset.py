import json
import os
import random

from torch import Tensor
from torch.utils.data import Dataset


class KWSDataset(Dataset):
    keywords = ["montserrat", "pedraforca", "vermell", "blau"]
    samples_folder = "./data/keywords"
    test_samples = 120

    def __init__(self, train: bool, transform=None):
        files = []
        for i, word in enumerate(self.keywords):
            file_list = os.listdir(f"{self.samples_folder}/{word}")
            random.shuffle(file_list)
            files.append(list(map(lambda f: f"{word}/{f}", file_list)))

        all_samples = list(sum(zip(*files), ()))

        begin = 0 if train else len(all_samples) - self.test_samples
        end = len(all_samples) - self.test_samples if train else len(all_samples)
        self.samples = all_samples[begin:end] if train else all_samples[begin:end]

        self.transform = transform

        # all_features = []
        # for i in range(0, self.__len__()):
        #     data = self.__get_raw_values__(i)
        #     all_features.append(data) # Remove batch dim
        # all_features_tensor = torch.stack(all_features)
        #
        # self.mean = all_features_tensor.mean()
        # self.std = all_features_tensor.std()
        self.mean = -0.7344476580619812
        self.std = 2404.997314453125

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
