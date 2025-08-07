import argparse
import random
from argparse import Namespace

import torch
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms

from KWSDataset import KWSDataset
from Nets import MnistNet, KWSNet
from quantization import scale_weights, descale_weights, get_scale_range
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MFCC
from collections.abc import Callable
from typing import TypeAlias

BitsFn: TypeAlias = str | Callable[[int, int], int | None]


class Experiment:
    def __init__(self, args: Namespace):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.torch_state = torch.random.get_rng_state()
        self.numpy_state = np.random.get_state()
        self.random_state = random.getstate()

        self.args = args

        use_accel = not args.no_accel and torch.accelerator.is_available()
        if use_accel:
            self.device = torch.accelerator.current_accelerator()
        else:
            self.device = torch.device("cpu")

        # self.net = MnistNet
        # transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        # self.loss_fn = torch.nn.NLLLoss(reduction='sum')
        # self.accuracy_loss_experiment_samples = 2000
        # self.fl_experiment_samples_per_batch = 20
        # self.fl_experiment_batches = 30

        self.net = KWSNet
        self.dataset_transform = transforms.Compose([
            MFCC(n_mfcc=13, melkwargs={
                "n_mels": 32,
                "hop_length": 200  # Default 200, increases accuracy but requires a larger input layer
            }),
            lambda x: x.flatten(),
        ])
        self.train_dataset_iid = KWSDataset(True, transform=self.dataset_transform, iid=True)
        test_dataset = KWSDataset(False, transform=self.dataset_transform)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_loss_experiment_samples = 200
        self.samples_per_fl_batch = 4
        self.fl_batches = 50

        self.train_kwargs = {'batch_size': 1}
        test_kwargs = {'batch_size': 1000}
        if use_accel:
            accel_kwargs = {
                'num_workers': 1,
                'pin_memory': True
            }
            self.train_kwargs.update(accel_kwargs)
            test_kwargs.update(accel_kwargs)

        self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    def reset_state(self):
        torch.random.set_rng_state(self.torch_state)
        np.random.set_state(self.numpy_state)
        random.setstate(self.random_state)

    def train(self, model, rng, optimizer, dataset):
        train_loader = torch.utils.data.DataLoader(
            Subset(dataset, list(rng)),
            **self.train_kwargs
        )

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss_fn(output, target).item()  # sum up batch loss
                predicted = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += predicted.eq(target.view_as(predicted)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return accuracy

    def init_models(self, count, net_kwargs):
        models = []
        optimizers = []
        for model_idx in range(count):
            model = self.net(**net_kwargs).to(self.device)
            # optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
            models.append(model)
            optimizers.append(optimizer)
        return models, optimizers

    def quantize_and_restore_weights(self, original_weights, quantization_bits):
        batch_size = len(original_weights) if self.args.batch_size is None else self.args.batch_size
        restored_weights = []
        for weights in np.array_split(original_weights, range(batch_size, len(original_weights), batch_size)):
            min_w, max_w, scaled_weights = scale_weights(weights.tolist(), quantization_bits)
            restored_weights += descale_weights(scaled_weights, quantization_bits, min_w, max_w)
        return restored_weights

    def quantize_weights(self, weights, quantization_bits):
        batch_size = len(weights) if self.args.batch_size is None else self.args.batch_size
        quantized_weights = []
        for weights in np.array_split(weights, range(batch_size, len(weights), batch_size)):
            min_w, max_w, scaled_weights = scale_weights(weights.tolist(), quantization_bits)
            quantized_weights.extend(scaled_weights)
        return quantized_weights

    def do_fl(self, models, send_bits: int | None, receive_bits: int | None | str = 'unset'):
        if receive_bits == 'unset':
            receive_bits = send_bits

        if send_bits is not None:
            weights = [self.quantize_and_restore_weights(model.get_flat_weights(), send_bits) for model in models]
        else:
            weights = [model.get_flat_weights() for model in models]
        averaged_weights = np.sum(weights, axis=0) / len(models)
        if receive_bits is not None:
            averaged_weights = self.quantize_and_restore_weights(averaged_weights, receive_bits)

        for model in models: model.set_flat_weights(averaged_weights)

    def accuracy_loss_experiment(self):
        self.reset_state()

        models, optimizers = self.init_models(1, {'hl_size': 20})
        self.train(models[0], range(0, self.accuracy_loss_experiment_samples), optimizers[0], self.train_dataset_iid)
        # original_accuracy = self.test(models[0])

        original_weights = models[0].get_flat_weights()
        scaled_weight_bits = list(range(1, 10, 1))
        accuracies = []
        for bits in tqdm(scaled_weight_bits):
            restored_weights = self.quantize_and_restore_weights(original_weights, bits)
            models[0].set_flat_weights(restored_weights)
            accuracy = self.test(models[0])
            accuracies.append(accuracy)

        plt.figure()
        plt.plot(scaled_weight_bits, accuracies)
        plt.xlabel("Quantization bits")  # add X-axis label
        plt.ylabel("Accuracy")  # add Y-axis label
        plt.ylim(-5, 105)
        plt.title("Accuracy drop vs quantization bits")
        plt.savefig(f"plots/accuracy_loss.png")

    def batch_training(self, models, optimizers, send_bits_fn: BitsFn, dataset, do_fl: bool = True,
                       receive_bits_fn: BitsFn | str = 'unset'):
        if receive_bits_fn == 'unset': receive_bits_fn = send_bits_fn

        x = [0]
        accuracies = [sum(self.test(model) for model in models) / len(models)]
        for batch_index in tqdm(range(self.fl_batches), desc="Training samples batch"):
            start = batch_index * self.samples_per_fl_batch * len(models)
            for model_index in range(len(models)):
                first_sample = start + (self.samples_per_fl_batch * model_index)
                last_sample = start + (self.samples_per_fl_batch * (model_index + 1))
                self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index], dataset)

            x.append((batch_index + 1) * self.samples_per_fl_batch)
            if do_fl:
                self.do_fl(models, send_bits_fn(batch_index, self.fl_batches),
                           receive_bits_fn(batch_index, self.fl_batches))
                accuracy = self.test(models[0])
            else:
                accuracy = sum([self.test(model) for model in models]) / len(models)
            accuracies.append(accuracy)

        return x, accuracies

    def accuracy_vs_epochs_vs_quant_bits_experiment(self, num_models):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(top=100, bottom=15)
        plt.title("Accuracy vs epochs")

        options = [
            {'do_fl': False, 'bits': None, 'label': 'No FL', 'color': 'b'},
            {'do_fl': True, 'bits': None, 'label': 'No quant', 'color': 'r'},
            {'do_fl': True, 'bits': 16, 'label': '16 bits', 'color': 'c--'},
            {'do_fl': True, 'bits': 8, 'label': '8 bits', 'color': 'm'},
            {'do_fl': True, 'bits': 6, 'label': '6 bits', 'color': 'g'},
            {'do_fl': True, 'bits': 5, 'label': '5 bits', 'color': 'brown'},
            {'do_fl': True, 'bits': 4, 'label': '4 bits', 'color': 'y'},
            {'do_fl': True, 'bits': 3, 'label': '3 bits', 'color': 'limegreen'},
            {'do_fl': True, 'bits': 2, 'label': '2 bits', 'color': 'k'},
        ]

        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, lambda i, nb: case['bits'], self.train_dataset_iid, do_fl=case['do_fl'])
            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend(loc='center right')
        plt.savefig(f"plots/accuracy_vs_epochs_vs_quant_bits.png")

    def early_vs_late_quantization(self, num_models: int, low_bits: int, high_bits: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Early vs late quantization")

        options = [
            {'label': 'Fixed high quantization (6)', 'bits': lambda i, nb: low_bits, 'color': 'r--'},
            {'label': 'Fixed low quantization (8)', 'bits': lambda i, nb: high_bits, 'color': 'g--'},
            {'label': f"Early quantization ({low_bits} → {high_bits})", 'bits': lambda i, nb: low_bits if i <= nb / 3 else high_bits, 'color': 'm-'},
            {'label': f"Late quantization ({high_bits} → {low_bits})", 'bits': lambda i, nb: high_bits if i <= nb / 3 else low_bits, 'color': 'b-'},
        ]
        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, case['bits'], self.train_dataset_iid)
            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/early_vs_late_quantization.png")

    def quantized_weights_histogram_experiment(self, num_models):
        self.reset_state()
        models, optimizers = self.init_models(num_models, {'hl_size': 20})
        self.batch_training(models, optimizers, lambda i, nb: None, self.train_dataset_iid)

        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(9, 3))

        original_weights = models[0].get_flat_weights()
        for i, bits in enumerate([8, 6, 4]):
            weights = self.quantize_weights(original_weights, bits)
            axs[i].hist(weights, bins=int(get_scale_range(bits)[1]))

        plt.savefig(f"plots/quantized_weights_histogram.png")

    def asymmetric_quantization_experiment(self, num_models: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Symmetric vs Asymmetric quantization")

        options = [
            {'label': 'Symmetric quantization (8)', 'send_bits': 8, 'receive_bits': 8, 'color': 'r--'},
            {'label': 'Asymmetric quantization (8 → 6)', 'send_bits': 8, 'receive_bits': 6, 'color': 'g--'},
            {'label': 'Asymmetric quantization (6 → 8)', 'send_bits': 6, 'receive_bits': 8, 'color': 'b--'},
        ]
        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, lambda i, nb: case['send_bits'], self.train_dataset_iid,
                                                receive_bits_fn=lambda i, nb: case['receive_bits'])
            plt.plot(x, accuracies, case['color'], label=case['label'])
        plt.legend()
        plt.savefig(f"plots/asymmetric_quantization.png")

    def random_quantization_experiment(self, num_models: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Random quantization")

        options = [
            {'label': 'Fixed 8-bit quantization', 'bits': lambda i, nb: 8, 'color': 'r--'},
            {'label': 'Random quantization (5-8 bits)', 'bits': lambda i, nb: random.randint(5, 8), 'color': 'g'},
            {'label': 'Random quantization (6-8 bits)', 'bits': lambda i, nb: random.randint(6, 8), 'color': 'b'},
        ]
        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, case['bits'], self.train_dataset_iid)
            plt.plot(x, accuracies, case['color'], label=case['label'])
        plt.legend()
        plt.savefig(f"plots/random_quantization.png")

    def nn_size_experiment(self, num_models: int):
        plt.figure()
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Final accuracy depending on HL size and quantization bits")

        for y in range(0, 100, 5):
            plt.axhline(y=y, color="#EEE", zorder=1)

        quantization_bits = [16, 8, 6, 5]
        hl_sizes = [10, 15, 20, 25]

        width, x = 0.15, np.arange(len(quantization_bits))
        for i, hl_size in enumerate(hl_sizes):
            accuracies = []
            for bits in quantization_bits:
                self.reset_state()
                models, optimizers = self.init_models(num_models, {'hl_size': hl_size})
                _, acc = self.batch_training(models, optimizers, lambda i, nb: bits, self.train_dataset_iid)
                accuracies.append(acc[-1])
            plt.bar(x + (width * i), accuracies, width, label=f"{hl_size} neurons", zorder=2)

        plt.xticks(x + (width * len(hl_sizes) / 2), [f"{bits} bits" for bits in quantization_bits])

        plt.legend(loc='upper right')
        plt.savefig(f"plots/nn_size.png")

    def iid_vs_non_experiment(self, num_models: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(top=100, bottom=15)
        plt.title("Accuracy vs epochs")

        train_dataset_non_iid_2 = KWSDataset(True, transform=self.dataset_transform, iid=2)
        train_dataset_non_iid_3 = KWSDataset(True, transform=self.dataset_transform, iid=3)
        train_dataset_non_iid_4 = KWSDataset(True, transform=self.dataset_transform, iid=4)

        options = [
            {'dataset': self.train_dataset_iid, 'bits': None, 'label': 'IID', 'color': 'r--'},
            {'dataset': train_dataset_non_iid_2, 'bits': None, 'label': 'Non-IID 2', 'color': 'g'},
            {'dataset': train_dataset_non_iid_3, 'bits': None, 'label': 'Non-IID 3', 'color': 'b'},
            {'dataset': train_dataset_non_iid_4, 'bits': None, 'label': 'Non-IID 4', 'color': 'm'},
        ]

        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, lambda i, nb: 8, case['dataset'])
            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/iid_vs_non.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--no-accel', action='store_true', help='Disable accelerator')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=None, help='Weights batch size')
    args = parser.parse_args()

    experiment = Experiment(args)

    experiment.accuracy_loss_experiment()
    experiment.accuracy_vs_epochs_vs_quant_bits_experiment(3)
    experiment.early_vs_late_quantization(3, 6, 8)
    experiment.quantized_weights_histogram_experiment(3)
    experiment.asymmetric_quantization_experiment(3)
    experiment.random_quantization_experiment(3)
    experiment.nn_size_experiment(3)
    experiment.iid_vs_non_experiment(3)


if __name__ == '__main__':
    main()
