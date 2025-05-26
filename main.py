import argparse
import math
from argparse import Namespace

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from Net import Net
from quantization import scale_weights, descale_weights
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Experiment:
    def __init__(self, args: Namespace):
        self.args = args

        use_accel = not args.no_accel and torch.accelerator.is_available()
        if use_accel:
            self.device = torch.accelerator.current_accelerator()
        else:
            self.device = torch.device("cpu")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.lr = args.lr

        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        self.train_kwargs = {'batch_size': 1, 'shuffle': True}
        test_kwargs = {'batch_size': 1000, 'shuffle': True}
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
            loss = F.nll_loss(output, target)
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
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                predicted = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += predicted.eq(target.view_as(predicted)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return accuracy

    def init_models(self, count):
        models = []
        optimizers = []
        for model_idx in range(count):
            model = Net().to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
            models.append(model)
            optimizers.append(optimizer)
        return models, optimizers

    def quantize_and_restore_weights(self, original_weights, quantization_bits):
        restored_weights = []
        for weights in np.array_split(original_weights, self.args.batch_size):
            min_w, max_w, scaled_weights = scale_weights(weights.tolist(), quantization_bits)
            restored_weights += descale_weights(scaled_weights, quantization_bits, min_w, max_w)
        return restored_weights

    def accuracy_loss_experiment(self):
        models, optimizers = self.init_models(1)
        self.train(models[0], range(0, 2000), optimizers[0])

        original_weights = models[0].get_flat_weights()
        scaled_weight_bits = list(range(2, 10, 1))
        accuracies = []
        for bits in scaled_weight_bits:
            restored_weights = self.quantize_and_restore_weights(original_weights, bits)
            models[0].set_flat_weights(restored_weights)
            accuracy = self.test(models[0])
            accuracies.append(accuracy)

        plt.figure()
        plt.plot(scaled_weight_bits, accuracies)
        plt.xlabel("Quantization bits")  # add X-axis label
        plt.ylabel("Accuracy")  # add Y-axis label
        plt.ylim(0,100)
        plt.title("Accuracy vs quantization bits")
        plt.savefig("plots/accuracies.png")

    def fl_experiment(self, baseline: bool, quantization_bits: int):
        num_models = 3
        models, optimizers = self.init_models(num_models)
        samples_per_batch = 20
        x = [0]
        device_accuracies = [[self.test(model)] for model in models]

        for batch_index in tqdm(range(30), desc="Training samples batch"):
            start = batch_index * samples_per_batch * num_models
            for model_index in range(num_models):
                first_sample = start + (samples_per_batch * model_index)
                last_sample = start + (samples_per_batch * (model_index + 1))
                self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index])
                device_accuracies[model_index].append(self.test(models[model_index]))

            x.append((batch_index + 1) * samples_per_batch)

            if not baseline:
                # Get the weights from model 1, quantize and restore them, merge them with model 0 and set them on model 0
                weights = [self.quantize_and_restore_weights(model.get_flat_weights(), quantization_bits) for model in models]
                averaged_weights = np.sum(weights, axis=0) / len(models)
                for model in models:
                    model.set_flat_weights(averaged_weights)

        plt.figure()
        for accuracy in device_accuracies:
            plt.plot(x, accuracy)
        plt.xlabel("Epoch")  # add X-axis label
        plt.ylabel("Accuracy")  # add Y-axis label
        plt.ylim(0, 100)
        plt.title("Accuracy vs epochs")
        plt.savefig(f"plots/accuracy_vs_epochs_{'baseline' if baseline else 'experiment'}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--no-accel', action='store_true', help='Disable accelerator')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=200, help='Weights batch size')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    experiment = Experiment(args)
    # experiment.accuracy_loss_experiment()
    # experiment.fl_experiment(True, 16)
    experiment.fl_experiment(False, 16)


if __name__ == '__main__':
    main()
