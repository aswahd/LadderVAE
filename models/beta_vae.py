import torch
from torch import nn
from torch.nn import functional as F


class BetaVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 bottleneck_size=1,
                 hidden_dims=None,
                 gamma=1.,
                 max_capacity: int = 25,
                 Capacity_max_iter=1e5,
                 loss_type='B',
                 **kwargs):
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.gamma = gamma
        self.in_channels = in_channels
        self.bottleneck_size = bottleneck_size
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.hidden_dims = hidden_dims
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * bottleneck_size ** 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * bottleneck_size ** 2, latent_dim)

        self.embedding = nn.Linear(num_classes, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * bottleneck_size ** 2)

        hidden_dims = hidden_dims[::-1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        var = F.softplus(var) + 1e-8

        return [mu, var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.bottleneck_size, self.bottleneck_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, var):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param var: (Tensor) Variance  of the latent Gaussian
        :return:
        """

        epsilon = torch.randn_like(mu)
        z = mu + (var ** 0.5) * epsilon
        return z

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def forward(self, x, y=None, test=False):

        mu, var = self.encode(x)

        z = mu
        if not test:
            z = self.reparameterize(mu, var)
        logits = self.classifier(z)
        # Class-conditional prior
        embedding_y = None
        if y is not None:
            embedding_y = self.embedding(y)
        x_rec = self.decode(z)
        return logits, embedding_y, mu, var, x_rec, *[None] * 5

    def compute_loss(self, *, x, y, logits, embedding_y, latent_mu, latent_var, x_rec, beta_kl, beta_entropy, **kwargs):
        assert embedding_y is not None, "embedding_y is None."
        # Reconstruction loss
        # rec_loss = F.mse_loss(x, x_rec)
        # rec_loss = F.binary_cross_entropy_with_logits(x_rec, torch.zeros_like(x_rec), reduction='none').sum(dim=(1, 2, 3)).mean()
        rec_loss = (x - x_rec).pow(2).sum(dim=(1, 2, 3)).mean()
        # KL Divergence loss
        kl_loss = 0.5 * (-torch.log(latent_var) + latent_var + (latent_mu - embedding_y).pow(2) - 1)
        kl_loss = kl_loss.sum(-1).mean()
        # Classification loss
        entropy_loss = F.cross_entropy(logits, y, label_smoothing=0.)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = rec_loss + kl_loss + entropy_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(x.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = rec_loss + self.gamma * (kl_loss - C).abs() + beta_entropy * entropy_loss
        else:
            raise ValueError('Undefined loss type.')

        return loss, rec_loss, kl_loss, entropy_loss

