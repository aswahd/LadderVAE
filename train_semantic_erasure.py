import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
from models import LadderVAE, BetaVAE, LatentLadderVAE
from torch import autograd
from mtl_gradient import pc_backward
from dataset import OODDataset


autograd.set_detect_anomaly(True)


class BetaScheduler:
    def __init__(self, max_steps=100, max_value=1., model='vae'):
        self.value = 0
        self.max_value = max_value
        self.increment = 1 / max_steps
        self.model = model

    def step(self):
        if self.model == 'vae':
            return self.max_value

        value = self.value + self.increment
        self.value = min(value, self.max_value)
        return self.value


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class Trainer:
    def __init__(self, model, optimizer, beta_scheduler, beta_entropy, num_classes, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta_scheduler = beta_scheduler
        self.beta_entropy = beta_entropy
        self.num_classes = num_classes

    def on_epoch_end(self):
        self.beta_scheduler.step()

    def train_step(self, x, y, x_out, y_out):
        self.model.train()
        self.optimizer.zero_grad()
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)

        # Forward pass
        # in-distribution
        logits, embedding_y, latent_mu, latent_var, x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list, mu = \
            self.model(x, y=y_onehot)
        
        # out-of-distribution
        logits_out, _, _, _, x_rec_out, *_  = self.model(x_out, y=None)[4]
        

        # Compute in-distribution loss
        loss, rec_loss, kl_loss, entropy_loss = self.model.compute_loss(x=x,
                                                                        y=y,
                                                                        logits=logits,
                                                                        embedding_y=embedding_y,
                                                                        latent_mu=latent_mu,
                                                                        latent_var=latent_var,
                                                                        x_rec=x_rec,
                                                                        dec_mu_list=dec_mu_list,
                                                                        dec_var_list=dec_var_list,
                                                                        q_mu_list=q_mu_list,
                                                                        q_var_list=q_var_list,
                                                                        beta_kl=self.beta_scheduler.value,
                                                                        beta_entropy=self.beta_entropy)
        

        # compute out-of-distribution loss
        # Maximize entropy of out-of-distribution data
        entropy_out = 0.5 * -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
        # Maximize reconstruction error of out-of-distribution data
        rec_loss_out = F.mse_loss(x_out, x_rec_out, reduction='none')
        if y_out is not None:
            # y_out is a mask
            # Weight the reconstruction loss by the mask
            # i.e., the foreground has higher weight and background has lower weight
            rec_loss_out = y_out * rec_loss_out
        rec_loss_out = rec_loss_out.sum(dim=(1,2,3)).mean(0)
        loss_out = entropy_out - rec_loss_out

        self.optimizer.zero_grad()

        loss += 0.1 * loss_out
        # pc_backward([rec_loss, kl_loss, entropy_loss, entropy_out, rec_loss_out], self.optimizer)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()

        return logits, mu, x_rec, loss.item(), rec_loss.item(), kl_loss.item(), entropy_loss.item()

    def val_step(self, x, y):
        self.model.eval()
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)

        # Forward pass
        with torch.no_grad():
            # Forward pass
            logits, embedding_y, latent_mu, latent_var, x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list, mu = \
                self.model(x, y=y_onehot, test=True)

            # Compute loss
            loss, rec_loss, kl_loss, entropy_loss = self.model.compute_loss(x=x,
                                                                            y=y,
                                                                            logits=logits,
                                                                            embedding_y=embedding_y,
                                                                            latent_mu=latent_mu,
                                                                            latent_var=latent_var,
                                                                            x_rec=x_rec,
                                                                            dec_mu_list=dec_mu_list,
                                                                            dec_var_list=dec_var_list,
                                                                            q_mu_list=q_mu_list,
                                                                            q_var_list=q_var_list,
                                                                            beta_kl=self.beta_scheduler.step(),
                                                                            beta_entropy=self.beta_entropy)

        loss = rec_loss + self.beta_entropy * entropy_loss + self.beta_scheduler.value * kl_loss

        return logits, mu, x_rec, loss.item(), rec_loss.item(), kl_loss.item(), entropy_loss.item()


class AverageMetrics:
    def __init__(self):
        self.correct = 0
        self.total_loss = 0
        self.rec_loss = 0
        self.kl_loss = 0
        self.entropy_loss = 0
        self.num_samples = 0

    def reset(self):
        self.__init__()

    def update(self, correct, total_loss, rec_loss, kl_loss, entropy_loss, num_samples):
        self.correct += correct
        self.total_loss += total_loss * num_samples
        self.rec_loss += rec_loss * num_samples
        self.kl_loss += kl_loss * num_samples
        self.entropy_loss += entropy_loss * num_samples
        self.num_samples += num_samples

    @property
    def accuracy(self):
        return 100. * self.correct / self.num_samples

    def get(self):
        return self.total_loss / self.num_samples, self.rec_loss / self.num_samples, self.kl_loss / self.num_samples, \
               self.entropy_loss / self.num_samples


def train(args, trainer, train_loader, val_loader, loader_ood):
    best_val_loss = float('inf')
    os.makedirs(f'checkpoints/{args.model}{args.beta_entropy}', exist_ok=True)
    with open(f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_train_loss.txt', 'w') as f:
        f.write('epoch, total_loss, rec_loss, kl_loss, entropy_loss\n')

    with open(f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_train_acc.txt', 'w') as f:
        f.write('epoch, accuracy\n')

    with open(f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_val_loss.txt', 'w') as f:
        f.write('epoch, total_loss, rec_loss, kl_loss, entropy_loss\n')
    with open(f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_val_acc.txt', 'w') as f:
        f.write('epoch, accuracy\n')

    train_metrics = AverageMetrics()
    val_metrics = AverageMetrics()

    # OOD data generator
    iter_ood = iter(loader_ood)
    for epoch in range(args.epochs):

        if epoch in args.lr_steps:
            trainer.optimizer.param_groups[0]['lr'] *= 0.1
            print("learning rate:", trainer.optimizer.param_groups[0]['lr'])
        train_metrics.reset()
        for batch_idx, (x, y) in enumerate(train_loader):

            # Get the next OOD data
            try:
                x_ood, y_ood = next(iter_ood)
            except StopIteration:
                iter_ood = iter(loader_ood)
                x_ood, y_ood = next(iter_ood)

            x, y = x.to(trainer.device), y.to(trainer.device)
            x_ood, y_ood = x_ood.to(trainer.device), y_ood.to(trainer.device)
            if not loader_ood.has_mask:
                # The reconstruction loss is equally maximized across the foreground
                # and the background
                y_ood = None

            # Train step
            logits, mu, x_rec, loss, rec_loss, kl_loss, entropy_loss = trainer.train_step(x, y, x_ood, y_ood)
            pred_labels = logits.argmax(-1)
            correct = torch.eq(pred_labels, y).sum().item()
            total = x.size(0)
            train_metrics.update(correct, loss, rec_loss, kl_loss, entropy_loss, total)
        trainer.on_epoch_end()
        # Log training stats
        with open(f'{args.model}{args.beta_entropy}/{args.dataset}_train_loss.txt', 'a') as f:
            loss, rec_loss, kl_loss, entropy_loss = train_metrics.get()
            f.write(f'{epoch}, {loss:4f}, {rec_loss:4f}, {kl_loss:4f}, {entropy_loss:4f}\n')

        # Log train accuracy
        with open(f'{args.model}{args.beta_entropy}/{args.dataset}_train_acc.txt', 'a') as f:
            f.write(f'{epoch},{train_metrics.accuracy:2f}\n')

        print('epoch: {}/{} train acc: {:.2f}%'.format(epoch, args.epochs, train_metrics.accuracy))

        # if args.val_every > 0 and epoch % args.val_every == 0:
        # Validation
        val_metrics.reset()
        for x, y in val_loader:
            x, y = x.to(trainer.device), y.to(trainer.device)
            val_logits, val_feats, x_rec, val_loss, val_rec_loss, val_kl_loss, val_entropy_loss = trainer.val_step(
                x, y)
            pred_labels = val_logits.argmax(-1)
            correct = torch.eq(pred_labels, y).sum().item()
            total = x.size(0)
            val_metrics.update(correct, val_loss, val_rec_loss, val_kl_loss, val_entropy_loss, total)

        # Log validation stats
        with open(f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_val_loss.txt', 'a') as f:
            loss, rec_loss, kl_loss, entropy_loss = val_metrics.get()
            f.write(f'checkpoints/{epoch}, {loss:4f}, {rec_loss:4f}, {kl_loss:4f}, {entropy_loss:4f}\n')

        with open(f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_val_acc.txt', 'a') as f:
            f.write(f'{epoch}, {val_metrics.accuracy:2f}\n')

        print(f'epoch: {epoch}/{args.epochs} val acc: {val_metrics.accuracy:.2f}%')

        # Save model
        ckpt = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "start_iter": epoch,
            "best_val_loss": best_val_loss
        }

        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(ckpt, f'checkpoints/{args.model}{args.beta_entropy}/{args.dataset}_best_ckpt.pt')


def main():
    parser = argparse.ArgumentParser(description='VAE Open Set Recognition')
    parser.add_argument('--dataset', default='cifar10', help='mnist|cifar10|cifar100')
    parser.add_argument('--model', default='llvae', help='lvae|vae')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.1, help='momentum (default: 1e-3)')
    parser.add_argument('--lr_steps', nargs='+', type=int, default=[60, 100, 150], help='LR steps')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--val_every', type=int, default=1, help='-1 to disable validation')
    parser.add_argument('--save_every', type=int, default=5, help='-1 to disable saving model')
    parser.add_argument('--beta_entropy', type=int, default=100, help='Cross-entropy loss weight')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    if args.dataset == 'cifar10':
        d = {'in_channels': 3, 'img_size': (32, 32), 'num_classes': 10}
        args = argparse.Namespace(**vars(args), **d)
        transform = transforms.Compose([
            # lambda x: x.convert('L'),  # grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        
        # in-distribution data
        train_dataset = datasets.CIFAR10('data/cifarpy', download=True, train=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = datasets.CIFAR10('data/cifarpy', download=False, train=False, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "cifar100":
        d = {'in_channels': 3, 'img_size': (32, 32), 'num_classes': 100}
        args = argparse.Namespace(**vars(args), **d)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.CIFAR100('data/cifarpy', download=True, train=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = datasets.CIFAR100('data/cifarpy', download=False, train=False, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "mnist":
        d = {'in_channels': 1, 'img_size': (32, 32), 'num_classes': 10}
        args = argparse.Namespace(**vars(args), **d)
        train_dataset = datasets.MNIST('data/mnist', download=True, train=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           lambda x: F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0),
                                           transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = datasets.MNIST('data/mnist', download=True, train=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         lambda x: F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=0),
                                         transforms.Normalize((0.1307,), (0.3081,))]))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
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


    # Out-of-distribution data
    dataset_ood = OODDataset(root=args.ood_path, transform=transforms)
    loader_ood = DataLoader(dataset_ood, args.batch_size, shuffle=True)

    print("Model number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)  #
    if args.model == 'vae':
        beta_scheduler = BetaScheduler(max_steps=50, max_value=1., model=args.model)
    elif args.model == 'lvae' or args.model == 'llvae':
        beta_scheduler = BetaScheduler(max_steps=50, max_value=1.,
                                       model=args.model)  # Linear warm-up from 0 to 1 over 50 epoch
    else:
        raise NotImplementedError("Beta scheduler not implemented")
    trainer = Trainer(model, optimizer, beta_scheduler, beta_entropy=args.beta_entropy, num_classes=args.num_classes,
                      device=device)
    train(args, trainer, train_loader, val_loader, loader_ood)


if __name__ == "__main__":
    main()
