import json
import os
import random
from torch import Tensor
from torch.utils.data import Dataset


class KWSDataset(Dataset):
    keywords = ["montserrat", "pedraforca", "vermell", "blau"]
    samples_folder = "./data/keywords"
    test_samples = 120
    samples_per_keyword = 180

    # iid: When 1, samples will be iid. Larger numbers use the same keyword as many times.
    def __init__(self, train: bool, transform=None, iid: int = 1):
        self.transform = transform

        num_files = iid

        all_samples = [None] * (len(self.keywords) * self.samples_per_keyword)
        for word_i, word in enumerate(self.keywords):
            file_list = os.listdir(f"{self.samples_folder}/{word}")
            random.shuffle(file_list)
            for file_i in range(0, len(file_list), num_files):
                for num_file in range(num_files):
                    index = (word_i * num_files) + (file_i * len(self.keywords)) + num_file
                    all_samples[index] = f"{word}/{file_list[file_i + num_file]}"

        begin = 0 if train else len(all_samples) - self.test_samples
        end = len(all_samples) - self.test_samples if train else len(all_samples)
        self.samples = all_samples[begin:end] if train else all_samples[begin:end]

        # all_features = []
        # for i in range(0, self.__len__()):
        #     data = self.__get_raw_values__(i)
        #     all_features.append(data) # Remove batch dim
        # all_features_tensor = torch.stack(all_features)
        self.mean = -0.7344476580619812  # all_features_tensor.mean()
        self.std = 2404.997314453125  # all_features_tensor.std()

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
