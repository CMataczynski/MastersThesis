import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import time
import datetime as dt
from utils import makedirs, RunningAverageMeter, inf_generator, learning_rate_with_decay
from ODEModel import NeuralODE, ResNet
from metrics import accuracy_and_f1, plot_confusion_matrix
from tqdm import tqdm


def create_experiment_name(experiment_name, is_odenet):
    if is_odenet:
        name = "ODE"
    else:
        name = "Resnet"
    name = experiment_name + "_" + str(dt.date.today()) + "_" + name

    max = 0
    for subdir in os.listdir(os.path.join(os.getcwd(), "experiments")):
        if name in subdir:
            var = int(subdir.split('_')[-1])
            if var > max:
                max = var
    return name + "_" + str(max+1)



def trainODE(experiment_name, dataset_loaders
            ,class_dict, training_length=150,
            is_odenet=True, batch_size=256, lr=0.1):
    '''
    experiment_name: string with name of the experiments
    dataset_loaders: (training loader, testing loader) with type of DataLoader

    '''
    no_classes = len(class_dict.keys())
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    name = create_experiment_name(experiment_name, is_odenet)
    writer = SummaryWriter(log_dir='experiments/' + str(name))
    makedirs(os.path.join(os.getcwd(), "experiments", name))

    if is_odenet:
        model = NeuralODE(no_classes).to(device)
    else:
        model = ResNet(no_classes).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    train_loader, test_loader = dataset_loaders
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(lr,
         batch_size, batch_denom=batch_size, batches_per_epoch=batches_per_epoch,
         boundary_epochs=[60, 100, 140],decay_rates=[1, 0.1, 0.01, 0.001]
         )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_acc = 0
    best_f1 = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    running_loss = 0.0
    for itr in tqdm(range(training_length * batches_per_epoch)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        model.train()
        optimizer.zero_grad()
        dct = data_gen.__next__()
        # print(dct)
        x = dct[0].float()
        y = dct[1]
        x = x.to(device)
        y = y.to(device)
        # x = x.unsqueeze(1)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = model.feature_layers[0].nfe
            model.feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if itr % 10 == 9:
            writer.add_scalar("Loss/train", running_loss / 10, itr)
            running_loss = 0.0
        if is_odenet:
            nfe_backward = model.feature_layers[0].nfe
            model.feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                model.eval()
                val_acc, f1 = accuracy_and_f1(model, test_loader, no_classes, device=device)
                if f1 > best_f1:
                    torch.save({'state_dict': model.state_dict()},
                                os.path.join(os.getcwd(),"experiments", name,
                                            'model_1.pth'))
                    best_f1 = f1
                    best_acc = val_acc
                writer.add_scalar("Accuracy/test", val_acc, itr//batches_per_epoch)
                writer.add_scalar("F1_score/test", f1, itr//batches_per_epoch)
                writer.add_scalar("NFE-F", f_nfe_meter.val, itr//batches_per_epoch)
                writer.add_scalar("NFE-B", b_nfe_meter.val, itr//batches_per_epoch)

    labs = []
    preds = []
    for data in test_loader:
        x = data[0].float().to(device)
        # x = x.unsqueeze(1)
        y = data[1].tolist()
        labs += y
        outputs = model(x)
        predicted = torch.max(outputs, 1).indices
        preds += predicted.tolist()

    labs = [class_dict[a] for a in labs]
    preds = [class_dict[a] for a in preds]
    writer.add_figure(name + " - Confusion Matrix",
                      plot_confusion_matrix(labs, preds,
                      [class_dict[key] for key in class_dict.keys()]))
    writer.close()
    return [best_acc, best_f1]
