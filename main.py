import pandas as pd
import random
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

BitsFn: TypeAlias = str | int | Callable[[int, int], int | None]

SEED = 2
SEEDS_COUNT = 15
LEARNING_RATE = 0.01
BATCH_SIZE = None


class Experiment:
    def __init__(self):
        self.set_seed(SEED)
        self.torch_state = None
        self.numpy_state = None
        self.random_state = None

        self.seeds = random.sample(range(1000), SEEDS_COUNT)

        use_accel = torch.accelerator.is_available()
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

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_loss_experiment_samples = 200
        self.samples_per_fl_batch = 4
        self.fl_batches = 50
        self.low_bits = 6
        self.high_bits = 8

        self.train_kwargs = {'batch_size': 1}
        self.test_kwargs = {'batch_size': 1000}
        if use_accel:
            accel_kwargs = {
                'num_workers': 1,
                'pin_memory': True
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.torch_state = torch.random.get_rng_state()
        self.numpy_state = np.random.get_state()
        self.random_state = random.getstate()

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

    def test(self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss_fn(output, target).item()  # sum up batch loss
                predicted = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += predicted.eq(target.view_as(predicted)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

    def init_models(self, count, net_kwargs):
        models = []
        optimizers = []
        for model_idx in range(count):
            model = self.net(**net_kwargs).to(self.device)
            # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=self.args.momentum)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            models.append(model)
            optimizers.append(optimizer)
        return models, optimizers

    def quantize_and_restore_weights(self, original_weights, quantization_bits):
        batch_size = len(original_weights) if BATCH_SIZE is None else BATCH_SIZE
        restored_weights = []
        for weights in np.array_split(original_weights, range(batch_size, len(original_weights), batch_size)):
            min_w, max_w, scaled_weights = scale_weights(weights.tolist(), quantization_bits)
            restored_weights += descale_weights(scaled_weights, quantization_bits, min_w, max_w)
        return restored_weights

    def quantize_weights(self, weights, quantization_bits):
        batch_size = len(weights) if BATCH_SIZE is None else BATCH_SIZE
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

    def batch_training(
            self, test_loader, models, optimizers, send_bits_fn: BitsFn, dataset, do_fl: bool = True,
            receive_bits_fn: BitsFn | str = 'unset'
    ):
        if receive_bits_fn == 'unset': receive_bits_fn = send_bits_fn

        x = [0]
        accuracies = [sum(self.test(model, test_loader) for model in models) / len(models)]
        for batch_index in tqdm(range(self.fl_batches), desc="Training samples batch"):
            start = batch_index * self.samples_per_fl_batch * len(models)
            for model_index in range(len(models)):
                first_sample = start + (self.samples_per_fl_batch * model_index)
                last_sample = start + (self.samples_per_fl_batch * (model_index + 1))
                self.train(models[model_index], range(first_sample, last_sample), optimizers[model_index], dataset)

            x.append((batch_index + 1) * self.samples_per_fl_batch)
            if do_fl:
                self.do_fl(
                    models,
                    send_bits_fn(batch_index, self.fl_batches) if callable(send_bits_fn) else send_bits_fn,
                    receive_bits_fn(batch_index, self.fl_batches) if callable(receive_bits_fn) else send_bits_fn
                )
                accuracy = self.test(models[0], test_loader)
            else:
                accuracy = sum([self.test(model, test_loader) for model in models]) / len(models)
            accuracies.append(accuracy)

        return x, accuracies

    def get_test_loader(self):
        test_dataset = KWSDataset(False, transform=self.dataset_transform)
        return torch.utils.data.DataLoader(test_dataset, **self.test_kwargs)

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
        ]

        excel_data = {}
        epochs = None

        for case in options:
            case_accuracies = []
            for seed in self.seeds:
                self.set_seed(seed)

                test_loader = self.get_test_loader()
                train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=True)

                models, optimizers = self.init_models(num_models, {'hl_size': 20})
                x, accuracies = self.batch_training(
                    test_loader, models, optimizers, case['bits'], train_dataset, do_fl=case['do_fl']
                )
                case_accuracies.append(accuracies)

            avg_acc = np.average(case_accuracies, axis=0)

            if epochs is None:
                epochs = x
                excel_data["Epoch"] = x

            excel_data[case['label']] = avg_acc
            plt.plot(x, avg_acc, case['color'], label=case['label'])

        plt.legend(loc='center right')
        plt.savefig("plots/accuracy_vs_epochs_vs_quant_bits.png")

        df = pd.DataFrame(excel_data)
        df.to_excel("plots/accuracy_vs_epochs_vs_quant_bits.xlsx", index=False)

    def early_vs_late_quantization(self, num_models: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(top=100, bottom=15)
        plt.title("Early vs late quantization")

        options = [
            {'label': 'Fixed high quantization (6)', 'bits': self.low_bits, 'color': 'r--'},
            {'label': 'Fixed low quantization (8)', 'bits': self.high_bits, 'color': 'g--'},
            {'label': f"Early quantization ({self.low_bits} → {self.high_bits})",
             'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits, 'color': 'm-'},
            {'label': f"Late quantization ({self.high_bits} → {self.low_bits})",
             'bits': lambda i, nb: self.high_bits if i <= nb / 3 else self.low_bits, 'color': 'b-'},
        ]
        for case in options:
            case_accuracies = []
            for seed in self.seeds:
                self.set_seed(seed)

                test_loader = self.get_test_loader()
                train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=True)

                models, optimizers = self.init_models(num_models, {'hl_size': 20})
                x, accuracies = self.batch_training(test_loader, models, optimizers, case['bits'], train_dataset)
                case_accuracies.append(accuracies)
            plt.plot(x, np.average(case_accuracies, axis=0), case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/early_vs_late_quantization.png")

    def quantized_weights_histogram_experiment(self, num_models):
        self.reset_state()

        test_loader = self.get_test_loader()
        train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=True)

        models, optimizers = self.init_models(num_models, {'hl_size': 20})
        self.batch_training(test_loader, models, optimizers, None, train_dataset)

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
            {'label': f"Symmetric quantization ({self.high_bits})", 'send_bits': self.high_bits,
             'receive_bits': self.high_bits, 'color': 'r--'},
            {'label': f"Symmetric quantization ({self.low_bits})", 'send_bits': self.low_bits,
             'receive_bits': self.high_bits, 'color': 'g--'},
            {'label': f"Asymmetric quantization ({self.high_bits} → {self.low_bits})", 'send_bits': self.high_bits,
             'receive_bits': self.low_bits, 'color': 'b'},
            {'label': f"Asymmetric quantization (6 → 8)", 'send_bits': self.low_bits, 'receive_bits': self.high_bits,
             'color': 'm'},
        ]
        for case in options:
            case_accuracies = []
            for seed in self.seeds:
                self.set_seed(seed)

                test_loader = self.get_test_loader()
                train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=True)

                models, optimizers = self.init_models(num_models, {'hl_size': 20})
                x, accuracies = self.batch_training(
                    test_loader, models, optimizers, case['send_bits'], train_dataset,
                    receive_bits_fn=case['receive_bits']
                )
                case_accuracies.append(accuracies)
            plt.plot(x, np.average(case_accuracies, axis=0), case['color'], label=case['label'])
        plt.legend()
        plt.savefig(f"plots/asymmetric_quantization.png")

    def random_quantization_experiment(self, num_models: int):
        plt.figure()
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)
        plt.title("Random quantization")
        # plt.xlabel("Epoch") # For line plot

        cases = [
            {'label': 'Fixed\n(8-bit)', 'bits': 8},  # , 'color': 'r--'},
            {'label': 'Random\n (6-8 bits)', 'bits': lambda i, nb: random.randint(6, 8)},  # , 'color': 'g'},
            {'label': 'Random\n (6-10 bits)', 'bits': lambda i, nb: random.randint(6, 10)},  # , 'color': 'b'},
            {'label': 'Random\n (6-12 bits)', 'bits': lambda i, nb: random.randint(6, 12)}  # , 'color': 'b'},
        ]
        y = []
        for case in cases:
            case_accuracies = []
            for seed in self.seeds:
                self.set_seed(seed)

                test_loader = self.get_test_loader()
                train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=True)

                models, optimizers = self.init_models(num_models, {'hl_size': 20})
                x, accuracies = self.batch_training(test_loader, models, optimizers, case['bits'], train_dataset)
                # plt.plot(x, accuracies, case['color'], label=case['label'])
                case_accuracies.append(accuracies[-1])
            y.append(np.average(case_accuracies))

        plt.bar(list(case['label'] for case in cases), y)  # For bar plot

        # plt.legend() # For line plot
        plt.savefig(f"plots/random_quantization.png")

    def nn_size_experiment(self, num_models: int):
        plt.figure(figsize=(7, 3))
        plt.subplots_adjust(bottom=0.18)
        plt.ylabel("Accuracy")
        plt.ylim(50, 100)
        plt.title("Final accuracy depending on HL size and quantization policy")

        for y in range(0, 100, 5):
            plt.axhline(y=y, color="#EEE", zorder=1)

        cases = [
            {'label': 'Static \n(8)', 'bits': 8},

            {'label': f"Early\n({self.low_bits} → {self.high_bits})",
             'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits},
            {'label': f"Late\n({self.high_bits} → {self.low_bits})",
             'bits': lambda i, nb: self.high_bits if i <= nb / 3 else self.low_bits},

            {'label': f"Asymmetric\n({self.high_bits} → {self.low_bits})",
             'bits': self.high_bits, 'receive_bits': self.low_bits},
            {'label': f"Asymmetric\n({self.low_bits} → {self.high_bits})",
             'bits': self.low_bits, 'receive_bits': self.high_bits},

            {'label': f"Random\n({self.low_bits} → {self.high_bits})",
             'bits': lambda i, nb: random.randint(6, 10)},
        ]

        hl_sizes = [10, 15, 20, 25]

        width = 0.15
        x = np.arange(len(cases))

        # ----------------- NEW -----------------
        excel_rows = []
        # ---------------------------------------

        for i, hl_size in enumerate(hl_sizes):
            accuracies = []

            for case in cases:
                case_accuracies = []
                for seed in self.seeds:
                    self.set_seed(seed)

                    test_loader = self.get_test_loader()
                    train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=True)

                    models, optimizers = self.init_models(num_models, {'hl_size': hl_size})
                    _, acc = self.batch_training(
                        test_loader,
                        models,
                        optimizers,
                        case['bits'],
                        train_dataset,
                        receive_bits_fn=case['receive_bits'] if 'receive_bits' in case else 'unset'
                    )
                    case_accuracies.append(acc[-1])

                mean_acc = np.average(case_accuracies)
                accuracies.append(mean_acc)

                # Save for Excel
                excel_rows.append({
                    "HL size": hl_size,
                    "Quantization policy": case['label'].replace("\n", " "),
                    "Final accuracy": mean_acc
                })

            plt.bar(x + (width * i), accuracies, width, label=f"{hl_size} neurons", zorder=2)

        plt.xticks(x + (width * len(hl_sizes) / 2), [f"{case['label']}" for case in cases])
        plt.legend(loc='lower left')
        plt.savefig("plots/nn_size.png")

        # -------- EXPORT ----------
        df = pd.DataFrame(excel_rows)
        df_pivot = df.pivot(index="Quantization policy", columns="HL size", values="Final accuracy")
        df_pivot.to_excel("plots/nn_size.xlsx")

    def iid_vs_non_experiment(self, num_models: int):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(top=100, bottom=15)
        plt.title("Accuracy vs epochs")

        cases = [
            {'iid': True, 'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits, 'label': 'Early IID',
             'color': 'r--'},
            {'iid': 2, 'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits,
             'label': 'Early non-IID 2', 'color': 'g'},
            {'iid': 3, 'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits,
             'label': 'Early non-IID 3', 'color': 'b'},
            {'iid': 4, 'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits,
             'label': 'Early non-IID 4', 'color': 'm'},
        ]

        for case in cases:
            case_accuracies = []
            for seed in self.seeds:
                self.set_seed(seed)

                test_loader = self.get_test_loader()
                train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=case['iid'])

                models, optimizers = self.init_models(num_models, {'hl_size': 20})
                x, accuracies = self.batch_training(test_loader, models, optimizers, case['bits'], train_dataset)
                case_accuracies.append(accuracies)
            plt.plot(x, np.average(case_accuracies, axis=0), case['color'], label=case['label'])

        plt.legend()
        plt.savefig(f"plots/iid_vs_non.png")

    def non_iid_policies_experiment(self, num_models: int):
        plt.figure(figsize=(7, 3))
        plt.subplots_adjust(bottom=0.18)
        plt.ylabel("Accuracy")
        plt.ylim(50, 100)
        plt.title("Final accuracy depending on IID degree and quantization policy")

        for y in range(0, 100, 5):
            plt.axhline(y=y, color="#EEE", zorder=1)

        cases = [
            {'label': 'Static \n(8)', 'bits': 8},

            {'label': f"Early\n({self.low_bits} → {self.high_bits})",
             'bits': lambda i, nb: self.low_bits if i <= nb / 3 else self.high_bits},
            {'label': f"Late\n({self.high_bits} → {self.low_bits})",
             'bits': lambda i, nb: self.high_bits if i <= nb / 3 else self.low_bits},

            {'label': f"Asymmetric\n({self.high_bits} → {self.low_bits})",
             'bits': self.high_bits, 'receive_bits': self.low_bits},
            {'label': f"Asymmetric\n({self.low_bits} → {self.high_bits})",
             'bits': self.low_bits, 'receive_bits': self.high_bits},

            {'label': f"Random\n({self.low_bits} → {self.high_bits})",
             'bits': lambda i, nb: random.randint(6, 10)},
        ]

        iid_policies = [True, 2, 3, 4]

        width = 0.15
        x = np.arange(len(cases))

        # -------- NEW --------
        excel_rows = []
        # --------------------

        for i, iid_policy in enumerate(iid_policies):
            accuracies = []

            for case in cases:
                case_accuracies = []
                for seed in self.seeds:
                    self.set_seed(seed)

                    test_loader = self.get_test_loader()
                    train_dataset = KWSDataset(True, transform=self.dataset_transform, iid_factor=iid_policy)

                    models, optimizers = self.init_models(num_models, {'hl_size': 20})
                    _, acc = self.batch_training(
                        test_loader,
                        models,
                        optimizers,
                        case['bits'],
                        train_dataset,
                        receive_bits_fn=case['receive_bits'] if 'receive_bits' in case else 'unset'
                    )
                    case_accuracies.append(acc[-1])

                mean_acc = np.average(case_accuracies)
                accuracies.append(mean_acc)

                excel_rows.append({
                    "IID policy": iid_policy,
                    "Quantization policy": case['label'].replace("\n", " "),
                    "Final accuracy": mean_acc
                })

            plt.bar(x + (width * i), accuracies, width, label=f"{iid_policy} IID", zorder=2)

        plt.xticks(x + (width * len(iid_policies) / 2), [f"{case['label']}" for case in cases])
        plt.legend(loc='lower left')
        plt.savefig("plots/non_iid_policies_experiment.png")

        # ------- EXPORT -------
        df = pd.DataFrame(excel_rows)
        df_pivot = df.pivot(index="Quantization policy", columns="IID policy", values="Final accuracy")
        df_pivot.to_excel("plots/non_iid_policies_experiment.xlsx")


def main():
    experiment = Experiment()

    # experiment.accuracy_loss_experiment()
    # experiment.accuracy_vs_epochs_vs_quant_bits_experiment(3)
    # experiment.early_vs_late_quantization(3)
    # experiment.quantized_weights_histogram_experiment(3)
    # experiment.asymmetric_quantization_experiment(3)
    # experiment.random_quantization_experiment(3)
    # experiment.nn_size_experiment(3)
    # experiment.iid_vs_non_experiment(3)
    experiment.non_iid_policies_experiment(3)


if __name__ == '__main__':
    main()
