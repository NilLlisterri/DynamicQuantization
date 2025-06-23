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
        self.fl_experiment_samples_per_batch = 4
        self.fl_experiment_batches = 50

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

    def accuracy_loss_experiment(self, filename: str):
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

    def fl_experiment(self, fl: bool, quantization_bits: int | None, filename):
        num_models = 3
        models, optimizers = self.init_models(num_models)
        samples_per_batch = self.fl_experiment_samples_per_batch
        x = [0]
        device_accuracies = [[self.test(model)] for model in models]

        for batch_index in tqdm(range(self.fl_experiment_batches), desc="Training samples batch"):
            start = batch_index * samples_per_batch * num_models
            for model_index in range(num_models):
                first_sample = start + (samples_per_batch * model_index)
                last_sample = start + (samples_per_batch * (model_index + 1))
                self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index])
                device_accuracies[model_index].append(self.test(models[model_index]))

            x.append((batch_index + 1) * samples_per_batch)

            if fl:
                if quantization_bits is not None:
                    weights = [
                        self.quantize_and_restore_weights(model.get_flat_weights(), quantization_bits) for model in
                        models
                    ]
                else:
                    weights = [
                        model.get_flat_weights() for model in models
                    ]
                averaged_weights = np.sum(weights, axis=0) / len(models)
                if quantization_bits is not None:
                    averaged_weights = self.quantize_and_restore_weights(averaged_weights, quantization_bits)
                for model in models:
                    model.set_flat_weights(averaged_weights)

        plt.figure()
        for accuracy in device_accuracies:
            plt.plot(x, accuracy)
        plt.xlabel("Epoch")  # add X-axis label
        plt.ylabel("Accuracy")  # add Y-axis label
        plt.ylim(0, 100)
        plt.title("Accuracy vs epochs")
        plt.savefig(f"plots/{filename}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--no-accel', action='store_true', help='Disable accelerator')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=None, help='Weights batch size')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    def reset_state():
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        random.setstate(random_state)

    experiment = Experiment(args)

    experiment.accuracy_loss_experiment('accuracy_vs_quantization_bits')

    reset_state()
    experiment.fl_experiment(False, None, 'accuracy_vs_epochs_no_fl')
    reset_state()
    experiment.fl_experiment(True, None, 'accuracy_vs_epochs_no_quantization')
    reset_state()
    experiment.fl_experiment(True, 32, 'accuracy_vs_epochs_32')
    reset_state()
    experiment.fl_experiment(True, 16, 'accuracy_vs_epochs_16')
    reset_state()
    experiment.fl_experiment(True, 8, 'accuracy_vs_epochs_8')
    reset_state()
    experiment.fl_experiment(True, 4, 'accuracy_vs_epochs_4')
    reset_state()
    experiment.fl_experiment(True, 2, 'accuracy_vs_epochs_2')
    reset_state()
    experiment.fl_experiment(True, 1, 'accuracy_vs_epochs_1')


if __name__ == '__main__':
    main()
