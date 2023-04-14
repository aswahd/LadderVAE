import torch
from . import utils
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ks, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ks, stride, padding),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, padding, latent_dim, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.out_channels = out_channels
        self.conv = ConvBlock(in_channels, out_channels, ks, stride, padding)
        self.fc_mean = nn.Linear(out_channels * img_size[0] * img_size[1], self.latent_dim)
        self.fc_var = nn.Linear(out_channels * img_size[0] * img_size[1], self.latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h_flat = h.view(-1, self.out_channels * self.img_size[0] * self.img_size[1])
        mu, var = self.fc_mean(h_flat), self.fc_var(h_flat)
        var = F.softplus(var) + 1e-8
        return h, mu, var


class Encoder(nn.Module):
    def __init__(self, dims, latent_dim, img_size, num_classes):
        super().__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes
        self.encoder = nn.ModuleList()
        _h, _w = img_size
        for i in range(len(dims) - 1):
            _h, _w = _h // 2, _w // 2
            in_channels = dims[i]
            out_channels = dims[i + 1]
            latent = latent_dim[i]
            block = EncoderBlock(in_channels, out_channels, 3, 2, 1, latent, (_h, _w))
            self.encoder.append(block)

    def forward(self, x):
        h, mu, var = [], [], []
        h_ = x
        for block in self.encoder:
            h_, mu_, var_ = block(h_)
            # print(h_.shape, mu_.shape, var_.shape)
            h.append(h_)
            mu.append(mu_)
            var.append(var_)
        return h, mu, var


class ConvTransposeBlock(nn.Module):
    def __init__(self,  in_latent, in_channels, out_channels, ks, stride, padding, img_size, act='prelu', final_layer=False):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.fc = nn.Linear(in_latent, in_channels * img_size[0] * img_size[1])
        self.prelu = nn.PReLU()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, ks, stride, padding),  # (w-k+2p)/s+1
            nn.BatchNorm2d(out_channels, affine=False) if not final_layer else nn.Identity(),
            nn.PReLU() if act == 'prelu' else nn.Tanh(),
        )

    def forward(self, z):
        x = self.prelu(self.fc(z))
        x = x.view(-1, self.in_channels, self.img_size[0], self.img_size[1])
        h = self.conv(x)  # 2x
        return h


class DecoderBlock(nn.Module):
    def __init__(self, in_latent, out_latent, in_channels, out_channels, ks, stride, padding, img_size, act='prelu'):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        # self.fc = nn.Linear(in_latent, out_channels * img_size[0] * img_size[1])
        self.prelu = nn.PReLU()
        self.conv = ConvTransposeBlock(in_latent, in_channels, out_channels, ks, stride, padding, img_size, act)
        self.fc_mean = nn.Linear(out_channels * 2 * img_size[0] * 2 * img_size[1], out_latent)   # Because conv tranpose is 2x
        self.fc_var = nn.Linear(out_channels * 2 * img_size[0] * 2 * img_size[1], out_latent)

    def forward(self, z):
        h = self.conv(z)
        h_flat = h.view(-1, torch.prod(torch.tensor(h.shape[1:])))
        mu, var = self.fc_mean(h_flat), self.fc_var(h_flat)
        var = F.softplus(var) + 1e-8
        return h, mu, var


class Decoder(nn.Module):
    def __init__(self, latent, channels, img_size):
        super().__init__()
        self.decoder = nn.ModuleList()
        _h, _w = img_size
        for i in range(len(channels) - 2):
            stride = 2
            ks = 2
            in_channels = channels[i]
            out_channels = channels[i + 1]
            in_latent = latent[i]
            out_latent = latent[i + 1]
            block = DecoderBlock(in_latent=in_latent,
                                 out_latent=out_latent,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 ks=ks,
                                 stride=stride,
                                 padding=0,
                                 img_size=(_h, _w),
                                 act='prelu'
                                 )
            _h, _w = _h * 2, _w * 2
            self.decoder.append(block)
        # the final decoder doesn't output mean and var
        block = ConvTransposeBlock(in_latent=latent[-1],
                                   in_channels=channels[-2],
                                   out_channels=channels[-1],
                                   ks=2,
                                   stride=2,
                                   padding=0,
                                   img_size=(_h, _w),
                                   act='tanh',
                                   final_layer=True
                                   )
        self.decoder.append(block)

    def forward(self, h, mu, var, bottle_neck):
        # h: [h1, h2, ...]
        # mu: [mu1, mu2, ...]
        # var: [var1, var2, ...]
        h, mu, var = h[::-1], mu[::-1], var[::-1]
        dec_h_list, dec_mu_list, dec_var_list = [], [], []
        q_mu_list, q_var_list = [], []
        # print("Decoder shape")
        for i, (_h, _mu, _var) in enumerate(zip(h, mu, var)):
            if i == 0:
                # sampled from the encoder
                z = bottle_neck
            else:
                # Sample latent
                z = utils.sample_gaussian(q_mu, q_var)

            # Decode latent
            dec_h, dec_mu, dec_var = self.decoder[i](z)
            # print(dec_h.shape, dec_mu.shape, dec_var.shape)
            # Lateral connection
            prec_enc = 1 / _var
            prec_dec = 1 / dec_var
            q_mu = (_mu * prec_enc + dec_mu * prec_dec) / (prec_enc + prec_dec)
            q_var = 1 / (prec_enc + prec_dec)

            # For loss calculation
            dec_h_list.append(dec_h)
            dec_mu_list.append(dec_mu)
            dec_var_list.append(dec_var)
            q_mu_list.append(q_mu)
            q_var_list.append(q_var)

        # Decoder final layer
        z = utils.sample_gaussian(q_mu, q_var)
        x_rec = self.decoder[-1](z)
        return x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list


class LadderVAE(nn.Module):

    def __init__(self, in_channels, img_size, num_classes):
        super().__init__()
        # channels = [in_channels] + 2 * [64] + 2 * [128] + 2 * [256] + 2 * [512] + 2 * [512]
        # latents =                  2 * [512] + 2 * [256] + 2 * [128] + 2 * [64] + 2 * [32]

        channels = [in_channels] + [64, 128, 256, 512, 512]
        latents =                  [64, 128, 256, 512, 512]

        self.encoder = Encoder(channels, latents, img_size, num_classes)
        downsampling = 2 ** len(latents)
        img_size = img_size[0]//downsampling, img_size[1]//downsampling
        self.decoder = Decoder(latents[::-1], channels[::-1], img_size)

        self.classifier = nn.Linear(latents[-1], num_classes)
        self.embedding = nn.Linear(num_classes, latents[-1])

        self.rec_criterion = nn.MSELoss()

    def forward(self, x, y=None, test=False):

        # encode
        h, mu, var = self.encoder(x)
        # bottleneck (classification)
        latent_mu, latent_var = mu[-1], var[-1]
        h, mu, var = h[:-1], mu[:-1], var[:-1]  # remove bottleneck
        z = latent_mu
        if not test:
            z = utils.sample_gaussian(latent_mu, latent_var)
        logits = self.classifier(z)
        # Class-conditional prior
        embedding_y = None
        if y is not None:
            embedding_y = self.embedding(y)

        # decode
        x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list = self.decoder(h, mu, var, z)

        return logits, embedding_y, latent_mu, latent_var, x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list, latent_mu

    @staticmethod
    def compute_loss(x, y, logits, embedding_y, latent_mu, latent_var, x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list,
                     beta_kl, beta_entropy):
        assert embedding_y is not None, "embedding_y is None."
        # Reconstruction loss
        rec_loss = (x - x_rec).pow(2).sum(dim=(1, 2, 3)).mean()
        # KL loss with class-conditional prior
        prior_mu, prior_var = torch.zeros_like(embedding_y), torch.ones_like(embedding_y)
        kl_loss = utils.kl_normal(latent_mu, latent_var, prior_mu, prior_var, embedding_y)
        for dec_mu, dec_var, q_mu, q_var in zip(dec_mu_list, dec_var_list, q_mu_list, q_var_list):
            kl_loss += utils.kl_normal(q_mu, q_var, dec_mu, dec_var, 0)
        kl_loss = torch.mean(kl_loss)
        # Classification loss
        entropy_loss = F.cross_entropy(logits, y)
        loss = rec_loss + beta_entropy * entropy_loss + beta_kl * kl_loss

        return loss, rec_loss, kl_loss, entropy_loss


if __name__ == "__main__":
    x = torch.rand(16, 1, 64, 64).cuda()
    y = torch.randint(0, 10, (16,)).cuda()
    y = F.one_hot(y, num_classes=10).float().cuda()
    model = LadderVAE(1, (64, 64), 10).cuda()
    logits, embedding_y, latent_mu, latent_var, x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list = model(x, y=y)
    rec_loss, kl_loss, entropy_loss = model.compute_loss(x, y, logits, embedding_y, latent_mu, latent_var, x_rec, dec_mu_list, dec_var_list, q_mu_list, q_var_list)
    loss = rec_loss + kl_loss + entropy_loss
    loss.backward()
    print(loss.item())
