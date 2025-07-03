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
from quantization import scale_weights, descale_weights
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MFCC


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

    def init_models(self, count):
        models = []
        optimizers = []
        for model_idx in range(count):
            model = self.net().to(self.device)
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

    def do_fl(self, models, bits: int | None):
        if bits is not None:
            weights = [self.quantize_and_restore_weights(model.get_flat_weights(), bits) for model in models]
        else:
            weights = [model.get_flat_weights() for model in models]
        averaged_weights = np.sum(weights, axis=0) / len(models)
        if bits is not None:
            averaged_weights = self.quantize_and_restore_weights(averaged_weights, bits)

        for model in models: model.set_flat_weights(averaged_weights)

    def accuracy_loss_experiment(self, filename: str):
        self.reset_state()

        models, optimizers = self.init_models(1)
        self.train(models[0], range(0, self.accuracy_loss_experiment_samples), optimizers[0])
        original_accuracy = self.test(models[0])

        original_weights = models[0].get_flat_weights()
        scaled_weight_bits = list(range(2, 10, 1))
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
        plt.savefig(f"plots/{filename}.png")
        print(f"Generated plots/{filename}.png")

    def accuracy_vs_epochs_vs_quant_bits_experiment(self, num_models):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Accuracy vs epochs")

        options = [
            {'fl': False, 'quantization_bits': None, 'label': 'No FL', 'color': 'b'},
            {'fl': True, 'quantization_bits': None, 'label': 'FL no quantization', 'color': 'r'},
            {'fl': True, 'quantization_bits': 16, 'label': '16 bits', 'color': 'c--'},
            {'fl': True, 'quantization_bits': 8, 'label': '8 bits', 'color': 'm'},
            {'fl': True, 'quantization_bits': 6, 'label': '6 bits', 'color': 'g'},
            {'fl': True, 'quantization_bits': 5, 'label': '5 bits', 'color': 'brown'},
            {'fl': True, 'quantization_bits': 4, 'label': '4 bits', 'color': 'y'},
            {'fl': True, 'quantization_bits': 3, 'label': '3 bits', 'color': 'limegreen'},
            {'fl': True, 'quantization_bits': 2, 'label': '2 bits', 'color': 'k'},
        ]

        for case in options:
            self.reset_state()

            models, optimizers = self.init_models(num_models)
            x = [0]
            accuracies = [sum(self.test(model) for model in models) / len(models)]

            for batch_index in tqdm(range(self.fl_batches), desc="Training samples batch"):
                start = batch_index * self.samples_per_fl_batch * num_models
                for model_index in range(num_models):
                    first_sample = start + (self.samples_per_fl_batch * model_index)
                    last_sample = start + (self.samples_per_fl_batch * (model_index + 1))
                    self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index])

                x.append((batch_index + 1) * self.samples_per_fl_batch)
                if case['fl']: self.do_fl(models, case['quantization_bits'])
                accuracies.append(self.test(models[0]))

            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/accuracy_vs_epochs_vs_quant_bits_experiment.png")

    def early_vs_late_quantization(self, num_models: int, low_bits: int, high_bits: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Early vs late quantization")

        options = [
            {'label': 'Fixed low quantization', 'bits': lambda i: low_bits, 'color': 'r--'},
            {'label': 'Fixed high quantization', 'bits': lambda i: high_bits, 'color': 'g--'},
            {'label': 'Early quantization', 'bits': lambda i: high_bits if i <= self.fl_batches / 3 else low_bits, 'color': 'b-'},
            {'label': 'Late quantization', 'bits': lambda i: low_bits if i <= self.fl_batches / 3 else high_bits, 'color': 'm-'},
        ]
        for case in options:
            self.reset_state()

            models, optimizers = self.init_models(num_models)
            samples_per_batch = self.samples_per_fl_batch
            x = [0]
            accuracies = [sum(self.test(model) for model in models) / len(models)]

            for batch_index in tqdm(range(self.fl_batches), desc="Training samples batch"):
                start = batch_index * samples_per_batch * num_models
                for model_index in range(num_models):
                    first_sample = start + (samples_per_batch * model_index)
                    last_sample = start + (samples_per_batch * (model_index + 1))
                    self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index])

                x.append((batch_index + 1) * samples_per_batch)
                bits = case['bits'](batch_index)
                self.do_fl(models, bits)
                accuracies.append(self.test(models[0]))

            plt.plot(x, accuracies, case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/early_vs_late_quantization.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--no-accel', action='store_true', help='Disable accelerator')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=None, help='Weights batch size')
    args = parser.parse_args()

    experiment = Experiment(args)

    # experiment.accuracy_loss_experiment('accuracy_vs_quantization_bits')
    #experiment.accuracy_vs_epochs_vs_quant_bits_experiment(3)
    experiment.early_vs_late_quantization(3, 6, 8)


if __name__ == '__main__':
    main()
