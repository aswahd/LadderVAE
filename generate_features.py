import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
from models import LadderVAE, LatentLadderVAE, BetaVAE
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Generate features for OOD detection')
parser.add_argument('--dataset', default='mnist', help='mnist|cifar10|cifar100')
parser.add_argument('--model', default='llvae', help='lvae|llvae|vae')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
parser.add_argument('--save_path', default='features', help='path to save the generated features and their labels.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.dataset == 'cifar10':
    d = {'in_channels': 3, 'img_size': (32, 32), 'num_classes': 10}
    args = argparse.Namespace(**vars(args), **d)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    in_data = datasets.CIFAR10('data/cifarpy', download=True, train=True, transform=transform)
    in_loader = DataLoader(in_data, batch_size=args.batch_size)

elif args.dataset == "cifar100":
    d = {'in_channels': 3, 'img_size': (32, 32), 'num_classes': 100}
    args = argparse.Namespace(**vars(args), **d)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    in_data = datasets.CIFAR100('data/cifarpy', download=True, train=True, transform=transform)
    in_loader = DataLoader(in_data, batch_size=args.batch_size)

elif args.dataset == "mnist":
    d = {'in_channels': 1, 'img_size': (32, 32), 'num_classes': 10}
    args = argparse.Namespace(**vars(args), **d)

    in_data = datasets.MNIST('data/mnist', download=True, train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 lambda x: F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0),
                                 transforms.Normalize((0.1307,), (0.3081,))]))
    in_loader = DataLoader(in_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
else:
    raise NotImplementedError("Dataset not implemented")


if args.model == 'vae':
    model = BetaVAE(in_channels=args.in_channels, num_classes=args.num_classes, latent_dim=512,
                    hidden_dims=[64, 128, 256, 512, 512],
                    bottleneck_size=1, loss_type='H')
elif args.model == 'lvae':
    model = LadderVAE(in_channels=args.in_channels, img_size=args.img_size, num_classes=args.num_classes)
elif args.model == "llvae":
    model = LatentLadderVAE(in_channels=args.in_channels, img_size=args.img_size, num_classes=args.num_classes)
else:
    raise NotImplementedError("Model not implemented")
ckpt = torch.load(args.checkpoint)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()


def get_features(loader):
    """ Save features and labels for correctly classified inputs. """
    model.eval()
    corr_features = []
    corr_labels = []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        y_onehot = F.one_hot(y, num_classes=args.num_classes).float().to(device)
        # Forward pass
        with torch.no_grad():
            # Forward pass
            logits, embedding_y, latent_mu, *_ = model(x, y=y_onehot, test=True)
            correct = torch.eq(torch.argmax(logits, dim=1), y)
            corr_features.append(latent_mu[correct].cpu())
            corr_labels.append(y[correct].cpu())

    return torch.cat(corr_features), torch.cat(corr_labels)


feats, labels = get_features(in_loader)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
torch.save(feats, f'{args.save_path}/{args.dataset}_train_features.pth')
torch.save(labels, f'{args.save_path}/{args.dataset}_train_labels.pth')
