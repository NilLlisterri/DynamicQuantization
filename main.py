import argparse
import random
from argparse import Namespace

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
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

BitsFn: TypeAlias = str | Callable[[int, int], int|None]

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
        transform = transforms.Compose([
            MFCC(n_mfcc=13, melkwargs={"n_mels": 32}),
            lambda x: x.flatten(),
        ])
        self.train_dataset = KWSDataset(True, transform=transform)
        test_dataset = KWSDataset(False, transform=transform)
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

    def train(self, model, rng, optimizer):
        train_loader = torch.utils.data.DataLoader(Subset(self.train_dataset, list(rng)),
                                                   **self.train_kwargs)  # Max 60k

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
        self.train(models[0], range(0, self.accuracy_loss_experiment_samples), optimizers[0])
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

    def batch_training(self, models, optimizers, send_bits_fn: BitsFn, do_fl: bool = True, receive_bits_fn: BitsFn|str = 'unset'):
        if receive_bits_fn == 'unset': receive_bits_fn = send_bits_fn

        x = [0]
        accuracies = [sum(self.test(model) for model in models) / len(models)]
        for batch_index in tqdm(range(self.fl_batches), desc="Training samples batch"):
            start = batch_index * self.samples_per_fl_batch * len(models)
            for model_index in range(len(models)):
                first_sample = start + (self.samples_per_fl_batch * model_index)
                last_sample = start + (self.samples_per_fl_batch * (model_index + 1))
                self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index])

            x.append((batch_index + 1) * self.samples_per_fl_batch)
            if do_fl: self.do_fl(models, send_bits_fn(batch_index, self.fl_batches), receive_bits_fn(batch_index, self.fl_batches))
            accuracies.append(self.test(models[0]))

        return x, accuracies

    def accuracy_vs_epochs_vs_quant_bits_experiment(self, num_models):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Accuracy vs epochs")

        options = [
            {'do_fl': False, 'bits': None, 'label': 'No FL', 'color': 'b'},
            {'do_fl': True, 'bits': None, 'label': 'FL no quantization', 'color': 'r'},
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
            x, accuracies = self.batch_training(models, optimizers, lambda i, nb: case['bits'], do_fl=case['do_fl'])
            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/accuracy_vs_epochs_vs_quant_bits.png")

    def early_vs_late_quantization(self, num_models: int, low_bits: int, high_bits: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Early vs late quantization")

        options = [
            {'label': 'Fixed low quantization', 'bits': lambda i, nb: low_bits, 'color': 'r--'},
            {'label': 'Fixed high quantization', 'bits': lambda i, nb: high_bits, 'color': 'g--'},
            {'label': 'Early quantization', 'bits': lambda i, nb: high_bits if i <= nb / 3 else low_bits, 'color': 'b-'},
            {'label': 'Late quantization', 'bits': lambda i, nb: low_bits if i <= nb / 3 else high_bits, 'color': 'm-'},
        ]
        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, case['bits'])
            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/early_vs_late_quantization.png")

    def quantized_weights_histogram_experiment(self, num_models):
        self.reset_state()
        models, optimizers = self.init_models(num_models, {'hl_size': 20})
        self.batch_training(models, optimizers, lambda i, nb: None)

        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(9, 3))

        original_weights = models[0].get_flat_weights()
        for i, bits in enumerate([8, 6, 4]):
            weights = self.quantize_weights(original_weights, bits)
            axs[i].hist(weights, bins=int(get_scale_range(bits)[1]))

        plt.savefig(f"plots/quantized_weights_histogram.png")

    def asymmetric_quantization_experiment(self, num_models):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Symmetric vs Asymmetric quantization")

        options = [
            {'label': 'Symmetric quantization', 'send_bits': 8, 'receive_bits': 8, 'color': 'r--'},
            {'label': 'Asymmetric quantization', 'send_bits': 8, 'receive_bits': 6, 'color': 'g--'},
        ]
        for case in options:
            self.reset_state()
            models, optimizers = self.init_models(num_models, {'hl_size': 20})
            x, accuracies = self.batch_training(models, optimizers, lambda i, nb: case['send_bits'], receive_bits_fn=lambda i, nb: case['receive_bits'])
            plt.plot(x, accuracies, case['color'], label=case['label'])
        plt.legend()
        plt.savefig(f"plots/asymmetric_quantization.png")


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

if __name__ == '__main__':
    main()
