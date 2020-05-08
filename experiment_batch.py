import torchvision
from training_loop import trainODE
from torchvision.transforms import Compose, ColorJitter, RandomAffine, RandomApply, RandomHorizontalFlip, Normalize, RandomCrop
import pandas as pd
import torch

batch_size = 128
df = pd.DataFrame({
    "name": [],
    "best_acc": [],
    "best_f1": []
})

experiments = [
    {
        "name": "CIFAR10_ODE",
        "is_odenet": True,
        "lr": 0.1,
        "train_dataset": torchvision.datasets.CIFAR10('datasets/cifar10',
                            transform=Compose([
                                RandomApply([
                                    ColorJitter(20,20,10,5)
                                ]),
                                RandomAffine(45, (0.2,0.2), (0.9,1.1)),
                                RandomHorizontalFlip(),
                                RandomCrop(32),
                                ToTensor(),
                                Normalize((0.4914, 0.4822, 0.4465), (0.2323, 0.1994, 0.2010))]),
                            train=True, download=True),
        "test_dataset": torchvision.datasets.CIFAR10('datasets/cifar10',
                            transform=transforms.ToTensor(),
                            train=False, download=True),
        "labels": {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }
    },
    {
        "name": "CIFAR10_ResNet",
        "is_odenet": False,
        "lr": 0.1,
        "train_dataset": torchvision.datasets.CIFAR10('datasets/cifar10',
                            transform=Compose([
                                RandomApply([
                                    ColorJitter(20,20,10,5)
                                ]),
                                RandomAffine(45, (0.2,0.2), (0.9,1.1)),
                                RandomHorizontalFlip(),
                                RandomCrop(32),
                                ToTensor(),
                                Normalize((0.4914, 0.4822, 0.4465), (0.2323, 0.1994, 0.2010))]),
                            train=True, download=True),
        "test_dataset": torchvision.datasets.CIFAR10('datasets/cifar10',
                            transform=torchvision.transforms.ToTensor(),
                            train=False, download=True),
        "labels": {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }
    },
    {
        "name": "MNIST_ODE",
        "is_odenet": True,
        "lr": 0.1,
        "train_dataset": torchvision.datasets.MNIST('datasets/mnist',
                            transform=Compose([
                                RandomAffine(45, (0.2,0.2), (0.9,1.1)),
                                RandomCrop(28),
                                ToTensor(),
                                Normalize((0.1307,), (0.3081,))]),
                            train=True, download=True),
        "test_dataset": torchvision.datasets.MNIST('datasets/mnist',
                            transform=torchvision.transforms.ToTensor(),
                            train=False, download=True),
        "labels": {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9"
        }
    },
    {
        "name": "MNIST_ResNet",
        "is_odenet": False,
        "lr": 0.1,
        "train_dataset": torchvision.datasets.MNIST('datasets/mnist',
                            transform=Compose([
                                RandomAffine(45, (0.2,0.2), (0.9,1.1)),
                                RandomCrop(28),
                                ToTensor(),
                                Normalize((0.1307,), (0.3081,))]),
                            train=True, download=True),
        "test_dataset": torchvision.datasets.MNIST('datasets/mnist',
                            transform=torchvision.transforms.ToTensor(),
                            train=False, download=True),
        "labels": {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9"
        }
    }
]

for experiment in experiments:
    best_acc, best_f1 = trainODE(experiment["name"],
                                 (torch.utils.data.DataLoader(experiment["train_dataset"],
                                                              batch_size=batch_size,
                                                              shuffle=True),
                                  torch.utils.data.DataLoader(experiment["test_dataset"],
                                                               batch_size=batch_size,
                                                               shuffle=True)),
                                 experiment["labels"],
                                 training_length=60,
                                 is_odenet=experiment["is_odenet"],
                                 batch_size=batch_size,
                                 lr=experiment["lr"])
    df_new = pd.DataFrame(
        {
        "name": [experiment["name"]],
        "best_acc": [best_acc],
        "best_f1": [best_f1]
        }
    )
    df = df.append(df_new, ignore_index=True)
    df.to_csv("results/results.csv")
