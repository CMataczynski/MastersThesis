import torchvision
from training_loop import trainODE
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
                            transform=torchvision.transforms.ToTensor(),
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
        "name": "CIFAR10_ResNet",
        "is_odenet": False,
        "lr": 0.1,
        "train_dataset": torchvision.datasets.CIFAR10('datasets/cifar10',
                            transform=torchvision.transforms.ToTensor(),
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
                            transform=torchvision.transforms.ToTensor(),
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
                            transform=torchvision.transforms.ToTensor(),
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
                                 training_length=150,
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
