import utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
import blitz.models.b_vgg as vgg
from blitz.utils import variational_estimator

MODE = 'cuda'  # ['cpu', 'cuda']
DIR = './data'
DATASET = 'LabelMe'  # ['LabelMe', 'CIFAR-10']
MODEL = 'NLE'  # ['SparseK', 'NL', 'NLE', 'JNN', 'Bag']
NUM_IN_ENSEMBLE = 3 # only for 'SparseK', 'NLE', 'Bag'


@variational_estimator
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(3, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(16*5*5, 120)
        self.fc2   = BayesianLinear(120, 84)
        self.fc3   = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_classifier(dataset, device):
    if dataset == 'LabelMe':
        return vgg.VGG(vgg.make_layers(vgg.cfg['A'], batch_norm=False), out_nodes=8).to(device)
    else:
        return BayesianCNN().to(device)


def adjust_learning_rate(initial, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = initial * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = utils.load_dataset(DIR, DATASET, MODEL)
    classifiers = []
    optimizers = []
    num_models = NUM_IN_ENSEMBLE if MODEL in ['SparseK', 'NLE', 'Bag'] else 1
    for i in range(num_models):
        classifier = get_classifier(DATASET, device)
        if MODE == 'cpu':
            classifier.cpu()
        else:
            classifier.cuda()

        if DATASET == 'LabelMe':
            lr = .0001
        elif DATASET == 'CIFAR-10':
            lr = .001
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        classifier.total_prob = 0
        classifiers.append(classifier)
        optimizers.append(optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    iteration = 0
    best_acc = 0
    nll_loss = torch.nn.CrossEntropyLoss()
    brier_score = torch.nn.MSELoss()
    torch_softmax = torch.nn.Softmax()
    for epoch in range(50):
        [adjust_learning_rate(lr, optimizers[i], epoch) for i in range(num_models)]
        for i, (datapoints, labels) in enumerate(train_loader):
            for m in range(num_models):
                optimizers[m].zero_grad()
                # ['SparseK', 'NL', 'NLE', 'JNN', 'Bag']
                complexity_cost_weight = 1/10000 if DATASET == 'LabelMe' else 1/50000
                if MODEL == 'SparseK' or MODEL == 'JNN':
                    loss = classifiers[m].sample_jnn_elbo(inputs=datapoints.to(device),
                                       labels=labels.to(device),
                                       criterion=criterion,
                                       sample_nbr=3,
                                       complexity_cost_weight=complexity_cost_weight,
                                       K=5
                    )
                elif MODEL == 'NL' or MODEL == 'NLE':
                    loss = classifiers[m].sample_noisy_elbo(inputs=datapoints.to(device),
                                       labels=labels.to(device),
                                       criterion=criterion,
                                       sample_nbr=3,
                                       complexity_cost_weight=complexity_cost_weight,
                    )
                elif MODEL == 'Bag':
                    sampling_mask = torch.randint(
                        high=8, size=(8,), dtype=torch.int64
                    )
                    sampling_mask = torch.unique(sampling_mask)  # remove duplicates
                    sampling_data = datapoints[sampling_mask]
                    sampling_target = labels[sampling_mask]
                    loss = classifiers[m].sample_noisy_elbo(inputs=sampling_data.to(device),
                                       labels=sampling_target.to(device),
                                       criterion=criterion,
                                       sample_nbr=3,
                                       complexity_cost_weight=complexity_cost_weight,
                    )                 
                loss.backward()
                optimizers[m].step()
                classifiers[m].total_prob /= 1.1
            iteration += 1
            if iteration%100==0:
                correct = 0
                total = 0
                loss_metric = 0
                brier_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        hard_labels = torch.distributions.Categorical(labels).sample()
                        outputs_list = []
                        total_probs = []
                        for m in range(num_models):
                            o = classifiers[m](images.to(device))
                            outputs_list.append(o)
                            try:
                                total_probs.append(classifiers[m].total_prob.item())
                            except:
                                total_probs.append(classifiers[m].total_prob)                        
                        softmaxes = softmax(total_probs)
                        outputs = outputs_list[0] * softmaxes[0]
                        for m in range(1, num_models):
                            outputs.add(outputs_list[m] * softmaxes[m])
                        loss_metric += nll_loss(outputs.to(device), hard_labels.to(device))
                        brier_loss += brier_score(torch_softmax(outputs).to(device), labels.to(device))
                        confidences, predicted = torch.max(outputs.data, 1)
                        total += hard_labels.size(0)
                        correct += (predicted == hard_labels.to(device)).sum().item()
                        accuracies = predicted.eq(hard_labels.to(device))
                print('Iteration: {} | Accuracy of the network on test images: {} %'.format(str(iteration) ,str(100 * correct / total)))
                length = len(test_loader.dataset)
                print('NLL: {}'.format(loss_metric / length))
                print('Brier: {}'.format(brier_loss / length))


if __name__ == '__main__':
    main()
