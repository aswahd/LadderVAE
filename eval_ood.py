from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import argparse
import os
import time
from models import LadderVAE, BetaVAE
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from omniglot import OmniglotLoader
import sklearn.metrics as sk

parser = argparse.ArgumentParser(description='VAE Open Set Recognition')
parser.add_argument('--dataset', default='mnist', help='mnist|cifar10|cifar100')
parser.add_argument('--model', default='vae', help='lvae|vae')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
parser.add_argument('--features_path', required=True, help='Path to load the generated features')
parser.add_argument('--results', default='results.txt', help='path to save the evaluation results.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
open(args.results, 'w').close()
if args.dataset == 'cifar10':
    d = {'in_channels': 3, 'img_size': (32, 32), 'num_classes': 10}
    args = argparse.Namespace(**vars(args), **d)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    in_data = datasets.CIFAR10('data/cifarpy', download=True, train=False, transform=transform)
    in_loader = DataLoader(in_data, batch_size=args.batch_size)

elif args.dataset == "cifar100":
    d = {'in_channels': 3, 'img_size': (32, 32), 'num_classes': 100}
    args = argparse.Namespace(**vars(args), **d)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    in_data = datasets.CIFAR100('data/cifarpy', download=True, train=False, transform=transform)
    in_loader = DataLoader(in_data, batch_size=args.batch_size)

elif args.dataset == "mnist":
    d = {'in_channels': 1, 'img_size': (32, 32), 'num_classes': 10}
    args = argparse.Namespace(**vars(args), **d)

    in_data = datasets.MNIST('data/mnist', download=True, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 lambda x: F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0),
                                 transforms.Normalize((0.1307,), (0.3081,))]))
    in_loader = DataLoader(in_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
else:
    raise NotImplementedError("Dataset not implemented")

if args.model == 'vae':
    model = BetaVAE(in_channels=args.in_channels, num_classes=args.num_classes, latent_dim=512, bottleneck_size=1,
                    loss_type='B')
elif args.model == 'lvae':
    model = LadderVAE(in_channels=args.in_channels, img_size=args.img_size, num_classes=args.num_classes)
else:
    raise NotImplementedError("Model not implemented")

ckpt = torch.load(args.checkpoint)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()

# load training features
train_features = torch.load(f'{args.features_path}/{args.dataset}_train_features.pth')
train_labels = torch.load(f'{args.features_path}/{args.dataset}_train_labels.pth')

# Build a multivariate Gaussian distribution from the training features
per_class_feats = [train_features[train_labels == i] for i in range(args.num_classes)]
# mu = map(np.mean, per_class_feats)
mu = [torch.mean(class_feats, dim=0) for class_feats in per_class_feats]
cov = [torch.from_numpy(np.cov(class_feats.numpy().T)) for class_feats in per_class_feats]
dists = [MultivariateNormal(m, c) for m, c in zip(mu, cov)]


def get_features(loader):
    """ Get the mean features from the encoder. """
    model.eval()
    feats = []
    for x, _ in tqdm(loader):
        x = x.to(device)
        with torch.no_grad():
            # Forward pass
            logits, _, mu, *_ = model(x, y=None, test=True)
            feats.append(mu.cpu())

    return torch.cat(feats)


def get_ood_scores(features, gaussian_dists):
    """ Calculate the log-likelihood of the data
        The maximum negative log-likelihood is considered as the OOD score.
        Args:
            features: (np.ndarray) shape (N, D)
            gaussian_dists: (list) of class-conditional multivariateNormal distribution
    """
    # scores = []
    scores = [dist.log_prob(features) for dist in gaussian_dists]
    scores = torch.stack(scores, dim=-1)  # (N, num_classes)
    scores = -torch.max(scores, dim=-1)[0]  # (N,)
    return scores


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr


def eval_ood(in_scores, out_scores, recall_level=0.95, ood_ds_name="", out_file='results.txt', make_plot=True):
    auroc, aupr, fpr = get_measures(out_scores, in_scores, recall_level)
    print(f"{ood_ds_name + ':':<20} auroc: {auroc:.4f}, aupr: {aupr:.4f}, fpr@{int(recall_level * 100):d}: {fpr:.4f}")
    with open(out_file, 'a') as f:
        f.write(
            f"{ood_ds_name + ':':<20} auroc: {auroc:.4f}, aupr: {aupr:.4f}, fpr@{int(recall_level * 100):d}: {fpr:.4f}\n")
    # Plot Histogram
    if make_plot:
        plt.figure(figsize=(5.5, 3), dpi=100)

        plt.title(f"Ladder VAE on {args.dataset} vs. {ood_ds_name} \n"
                  f" AUROC= {str(float(auroc * 100))[:6]}%", fontsize=14)

    vals, bins = np.histogram(out_scores, bins=100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="crimson", marker="", label="out test")
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="crimson", alpha=0.3)

    vals, bins = np.histogram(in_scores, bins=100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="navy", marker="", label="in test")
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="navy", alpha=0.3)

    if make_plot:
        plt.xlabel("OOD score (Negative Log Prob.)", fontsize=14)
        plt.ylabel("Count", fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0, None])

        plt.legend(fontsize=14)

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{args.dataset} vs. {ood_ds_name}_auroc.png")
        plt.close()


# Calculate the log-likelihood of the in-distribution data
in_features = get_features(in_loader)
in_scores = get_ood_scores(in_features, dists)

# Calculate the log-likelihood of the OOD data
# Omniglot
transform = transforms.Compose([lambda x: F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0)])
ood_loader = OmniglotLoader(batch_size=args.batch_size, train=False, transforms=transform, drop_last=False)
ood_features = get_features(ood_loader)
out_scores = get_ood_scores(ood_features, dists)
eval_ood(in_scores, out_scores, ood_ds_name="Omniglot")

# Fashion-MNIST
transform = transforms.Compose([transforms.ToTensor(),
                                lambda x: F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0),
                                transforms.Normalize((0.1307,), (0.3081,))])
ood_loader = datasets.FashionMNIST(root='data', train=False, download=False, transform=transform)
ood_loader = DataLoader(ood_loader, batch_size=args.batch_size, shuffle=False, drop_last=False)
ood_features = get_features(ood_loader)
out_scores = get_ood_scores(ood_features, dists)
eval_ood(in_scores, out_scores, ood_ds_name="Fashion-MNIST")


# Uniform Noise
size = 10_000
ood_data = torch.rand(size=(size, 3, 32, 32), dtype=torch.float32) * 2 - 1
ood_data = TensorDataset(ood_data, torch.zeros(size, dtype=torch.long))
ood_loader = DataLoader(ood_data, batch_size=args.batch_size)
ood_features = get_features(ood_loader)
out_scores = get_ood_scores(ood_features, dists)
eval_ood(in_scores, out_scores, ood_ds_name="Uniform Noise")

# Gaussian Noise
size = 10_000
ood_data = torch.randn(size=(size, 3, 32, 32), dtype=torch.float32)
ood_data = TensorDataset(ood_data, torch.zeros(size, dtype=torch.long))
ood_loader = DataLoader(ood_data, batch_size=args.batch_size)
ood_features = get_features(ood_loader)
out_scores = get_ood_scores(ood_features, dists)
eval_ood(in_scores, out_scores, ood_ds_name="Gaussian Noise")

# CIFAR100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
ood_data = datasets.CIFAR100('data/cifarpy', download=True, train=False, transform=transform)
ood_loader = DataLoader(ood_data, batch_size=args.batch_size, shuffle=True)
ood_features = get_features(ood_loader)
out_scores = get_ood_scores(ood_features, dists)
eval_ood(in_scores, out_scores, ood_ds_name="CIFAR100")
