import numpy as np
from sklearn.metrics import f1_score
from textwrap import wrap
import re
import itertools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def accuracy_and_f1(model, dataset_loader, device="cpu"):
    total_correct = 0
    labs = []
    preds = []
    for data in dataset_loader:
        x = data['image'].float().to(device)
        x = x.unsqueeze(1)
        y = one_hot(np.array(data['label'].numpy()), 5)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        labs += data['label'].tolist()
        preds += predicted_class.tolist()
        total_correct += np.sum(predicted_class == target_class)
    f1 = f1_score(labs, preds, average='weighted')
    return total_correct / len(dataset_loader.dataset), f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=False):
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    return fig
