import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        combined_input_dim = input_dim + condition_dim
        self.FC_input = nn.Linear(combined_input_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.batch_norm_input = nn.BatchNorm1d(input_dim, momentum=0.01, eps=0.001)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x, c):
        # x_norm = self.batch_norm_input(x)
        x_norm = self.batch_norm_input(x)
        x_c = torch.cat([x_norm, c], dim=1)
        _h = self.dropout(
            self.activation_fn(self.layer_norm(self.batch_norm(self.FC_input(x_c))))
        )
        h = self.dropout(
            self.activation_fn(
                self.layer_norm(
                    self.batch_norm(self.FC_hidden(torch.concat([_h, c], dim=1)))
                )
            )
        )
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_dim, condition_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        combined_latent_dim = latent_dim + condition_dim
        self.FC_latent = nn.Linear(combined_latent_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_output_mean = nn.Linear(hidden_dim, output_dim)
        self.FC_output_var = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        _h = self.dropout(
            self.activation_fn(self.layer_norm(self.batch_norm(self.FC_latent(x_c))))
        )
        h = self.dropout(
            self.activation_fn(
                self.layer_norm(
                    self.batch_norm(self.FC_hidden(torch.concat([_h, c], dim=1)))
                )
            )
        )
        x_hat = self.FC_output_mean(h)
        log_var_x_hat = self.FC_output_var(h)

        return x_hat, log_var_x_hat


class HVAE(pl.LightningModule):
    def __init__(self, Encoder, Decoder, nLeapFrog, LeapFrogStep, Temp0):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.nLeapFrog = nLeapFrog
        self.LeapFrogStep = LeapFrogStep
        self.Temp0 = (
            Temp0  # temp0 is just the standard deviation of auxiliary variables
        )

    def _his(self, z_0, p_0, x, c):
        _z = z_0
        _p = p_0
        for k in range(self.nLeapFrog):
            p_half = _p - 1.0 / 2 * self.LeapFrogStep * self._dU_dz(_z, x, c)
            _z = _z + self.LeapFrogStep * p_half
            _p = p_half - 1.0 / 2 * self.LeapFrogStep * self._dU_dz(_z, x, c)
        return _z, _p

    def _dU_dz(self, z, x, c):
        z = z.requires_grad_(True)

        x_hat, log_var_x_hat = self.Decoder(z, c)
        sigma_x = torch.exp(0.5 * log_var_x_hat)
        pxz = Independent(Normal(x_hat, sigma_x), 1)
        reconst_NLL = -pxz.log_prob(x)
        prior_NLL = 0.5 * torch.sum(z**2, dim=-1)

        U = (reconst_NLL + prior_NLL).sum()
        return torch.autograd.grad(U, z, create_graph=True)[0]

    def reparameterization(self, mean, sd):
        epsilon = torch.randn_like(sd)
        z = mean + sd * epsilon
        return z

    def forward(self, x, c):
        mean_z, log_var_z = self.Encoder(x, c)
        z_0 = self.reparameterization(mean_z, torch.exp(0.5 * log_var_z))
        p_0 = self.Temp0 * torch.randn_like(z_0)
        (z_k, p_k) = self._his(z_0, p_0, x, c)
        x_hat, log_var_x_hat = self.Decoder(z_k, c)
        return (x_hat, log_var_x_hat, mean_z, log_var_z, z_k, p_k)

    # ELBO
    def loss(self, x, x_hat, log_var_x_hat, mean_z, log_var_z, z_k, p_k):
        # reconstruction NLL
        sigma_x = torch.exp(0.5 * log_var_x_hat)
        pxz = Independent(Normal(x_hat, sigma_x), 1)
        NLL_x = -pxz.log_prob(x)  # shape: (batch,)

        # KL term between q(z|x) and p(z)
        latent_entropy_posterior = torch.sum(0.5 * log_var_z, dim=-1)
        NLL_z_prior = -0.5 * torch.sum(z_k * z_k, dim=-1)
        NLL_p_prior = -0.5 * torch.sum(p_k * p_k, dim=-1)

        return (
            NLL_x
            + latent_entropy_posterior
            + NLL_z_prior
            + NLL_p_prior
            + self.Encoder.latent_dim
        ).mean()

    def training_step(self, batch, batch_idx):
        x, c = batch
        x_hat, log_var_x_hat, mean_z, log_var_z, z_k, p_k = self.forward(x, c)
        loss = self.loss(x, x_hat, log_var_x_hat, mean_z, log_var_z, z_k, p_k)
        values = {"loss": loss}
        self.log_dict(values, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    pass
