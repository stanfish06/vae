import lightning as L
import pytorch_lightning as pl
import torch
import torch.nn as nn

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x):
        _h = self.dropout(
            self.layer_norm(self.batch_norm(self.activation_fn(self.FC_input(x))))
        )
        h = self.dropout(
            self.layer_norm(self.batch_norm(self.activation_fn(self.FC_hidden(_h))))
        )
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.FC_latent = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x):
        _h = self.dropout(
            self.layer_norm(self.batch_norm(self.activation_fn(self.FC_latent(x))))
        )
        h = self.dropout(
            self.layer_norm(self.batch_norm(self.activation_fn(self.FC_hidden(_h))))
        )
        x_hat = self.FC_output(h)
        log_var_x_hat = self.FC_output(h)

        return x_hat, log_var_x_hat


class VAE(L.LightningDataModule):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, sd):
        epsilon = torch.randn_like(sd)
        z = mean + sd * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat, log_var_x_hat = self.Decoder(z)
        return x_hat, log_var_x_hat, mean, log_var

    def loss(self, x, x_hat, log_var_x_hat, mean, log_var):
        from torch.distributions import Independent, Normal
        from torch.distributions.kl import kl_divergence

        sigma_x = torch.exp(0.5 * log_var_x_hat)
        pxz = Independent(Normal(x_hat, sigma_x), 1)
        reconst = -pxz.log_prob(x)  # shape: (batch,)

        # KL term between q(z|x) and p(z)
        sigma_z = torch.exp(0.5 * log_var)
        qzx = Independent(Normal(mean, sigma_z), 1)
        pz = Independent(Normal(torch.zeros_like(mean), torch.ones_like(sigma_z)), 1)
        kl = kl_divergence(qzx, pz)  # shape: (batch,)

        return (reconst + kl).mean()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, log_var_x_hat, mean, log_var = self.forward(x)
        loss = self.loss(x, x_hat, log_var_x_hat, mean, log_var)
        return loss

    def configure_optimizes(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    encoder = Encoder(input_dim=10, hidden_dim=50, latent_dim=5)
    decoder = Decoder(output_dim=10, hidden_dim=50, latent_dim=5)
    model = VAE(encoder, decoder)  # torch.
    trainer = pl.Trainer(accelerator="gpu")
    x = torch.randn(5, 10)
    x_hat, log_var_x_hat, mean, log_var = model.forward(x)
    print(x_hat.shape)
    print(log_var_x_hat.shape)
    print(mean.shape)
    print(log_var.shape)
    loss_test = model.loss(x, x_hat, log_var_x_hat, mean, log_var)
    print(loss_test)
