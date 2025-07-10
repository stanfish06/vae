import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Independent, NegativeBinomial

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.FC_input = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.batch_norm_input = nn.BatchNorm1d(input_dim, momentum=0.01, eps=0.001)

    def forward(self, x, c):
        # x_norm = self.batch_norm_input(x)
        x_norm = self.batch_norm_input(x)
        _h = self.dropout(
            self.activation_fn(
                self.batch_norm(self.FC_input(torch.concat([x_norm, c], dim=1)))
            )
        )
        h = self.dropout(
            self.activation_fn(
                self.batch_norm(self.FC_hidden(torch.concat([_h, c], dim=1)))
            )
        )
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_dim, condition_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.FC_latent = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_output_mean = nn.Linear(hidden_dim, output_dim)
        self.log_theta = nn.Parameter(torch.zeros(output_dim))
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)

    def forward(self, x, c):
        _h = self.dropout(
            self.activation_fn(
                self.batch_norm(self.FC_latent(torch.concat([x, c], dim=1)))
            )
        )
        h = self.dropout(
            self.activation_fn(
                self.batch_norm(self.FC_hidden(torch.concat([_h, c], dim=1)))
            )
        )
        log_x_hat = self.FC_output_mean(h)
        log_theta = self.log_theta.expand(log_x_hat.size(0), -1)

        return log_x_hat, log_theta


class HVAE(pl.LightningModule):
    def __init__(
        self, Encoder, Decoder, nLeapFrog, LeapFrogStep, Temp0, lr, beta, GraphRegWeight
    ):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.nLeapFrog = nLeapFrog
        self.LeapFrogStep = LeapFrogStep
        self.lr = lr
        self.beta = beta
        self.Temp0 = (
            Temp0  # temp0 is just the standard deviation of auxiliary variables
        )
        self.GraphRegWeight = GraphRegWeight

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

        log_x_hat, log_theta = self.Decoder(z, c)
        theta = torch.exp(log_theta)
        # Clamp for numerical stability
        log_x_hat = torch.clamp(log_x_hat, max=1e8)
        theta = torch.clamp(theta, min=1e-8, max=1e8)
        log_theta = torch.clamp(log_theta, max=1e8)
        # Parameterization for Negative Binomial
        # total_count = theta, probs = theta / (theta + x_hat)
        pxz = Independent(
            NegativeBinomial(total_count=theta, logits=log_x_hat - log_theta), 1
        )
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
        log_x_hat, log_theta = self.Decoder(z_k, c)
        return (log_x_hat, log_theta, mean_z, log_var_z, z_k, p_k)

    # ELBO
    def loss(
        self,
        x,
        log_x_hat,
        log_theta,
        mean_z,
        log_var_z,
        z_k,
        p_k,
        obs_idx,
        obs_nei_idx,
        obs_nei_conn,
    ):
        # reconstruction NLL
        theta = torch.exp(log_theta)
        # Clamp for numerical stability
        log_x_hat = torch.clamp(log_x_hat, max=1e8)
        theta = torch.clamp(theta, min=1e-8, max=1e8)
        log_theta = torch.clamp(log_theta, max=1e8)
        # Parameterization for Negative Binomial
        # total_count = theta, probs = theta / (theta + x_hat)
        pxz = Independent(
            NegativeBinomial(total_count=theta, logits=log_x_hat - log_theta), 1
        )
        NLL_x = -pxz.log_prob(x)  # shape: (batch,)

        # negative KL term between q(z|x) and p(z)
        latent_entropy_posterior = torch.sum(0.5 * log_var_z, dim=-1)
        NLL_z_prior = -0.5 * torch.sum(z_k * z_k, dim=-1)
        NLL_p_prior = -0.5 * torch.sum(p_k * p_k, dim=-1)

        # graph regularization
        gLoss = torch.tensor(0.0, device=z_k.device, dtype=z_k.dtype)
        # Create a mapping from global dataset index to local batch index
        global_to_local_idx = {
            int(global_idx): local_idx for local_idx, global_idx in enumerate(obs_idx)
        }
        n_pairs_in_batch = 0
        for i_local, i_global in enumerate(obs_idx):
            i_global_int = int(i_global)
            for j_global in obs_nei_idx[global_to_local_idx[i_global_int]]:
                # Check if the neighbor is also in the current batch
                # if j_global is -1, then obs i misses jth neghbor, so skip
                if j_global < 0:
                    continue
                if j_global in global_to_local_idx:
                    n_pairs_in_batch += 1
                    j_local = global_to_local_idx[j_global]
                    weight = obs_nei_conn[i_global_int, j_global]
                    gLoss += weight * torch.sum((z_k[i_local] - z_k[j_local]) ** 2)
        if n_pairs_in_batch > 0:
            gLoss = gLoss / n_pairs_in_batch
        gLoss = self.GraphRegWeight * gLoss

        return (
            NLL_x
            - self.beta
            * (
                latent_entropy_posterior
                + NLL_z_prior
                + NLL_p_prior
                + self.Encoder.latent_dim
            )
        ).mean() + gLoss

    def training_step(self, batch, batch_idx):
        x, c, obs_idx, obs_nei_idx, obs_nei_conn = batch
        log_x_hat, log_theta, mean_z, log_var_z, z_k, p_k = self.forward(x, c)
        loss = self.loss(
            x=x,
            log_x_hat=log_x_hat,
            log_theta=log_theta,
            mean_z=mean_z,
            log_var_z=log_var_z,
            z_k=z_k,
            p_k=p_k,
            obs_idx=obs_idx,
            obs_nei_idx=obs_nei_idx,
            obs_nei_conn=obs_nei_conn,
        )
        values = {"loss": loss}
        self.log_dict(values, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import numpy as np

    input_dim = 10
    condition_dim = 2
    hidden_dim = 20
    latent_dim = 5
    output_dim = 10
    n_obs = 10

    np.random.seed(42)
    x = torch.from_numpy(
        np.random.randint(0, 100, size=(n_obs, input_dim)).astype(np.float32)
    )
    np.random.seed(88)
    c = np.random.binomial(n=1, p=0.4, size=n_obs)
    c = torch.from_numpy(np.vstack([c, np.logical_not(c)]).T.astype(np.float32))

    # Dummy graph data
    obs_idx = torch.arange(n_obs)
    obs_nei_idx = {i: [] for i in range(n_obs)}
    obs_nei_conn = {}
    for i in range(n_obs - 1):
        # Connect node i and i+1
        obs_nei_idx[i].append(i + 1)
        obs_nei_idx[i + 1].append(i)
        # Add weights
        obs_nei_conn[i, i + 1] = 1.0
        obs_nei_conn[i + 1, i] = 1.0

    m = HVAE(
        Encoder=Encoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ),
        Decoder=Decoder(
            output_dim=output_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ),
        nLeapFrog=10,
        LeapFrogStep=0.01,
        Temp0=1,
        lr=1e-4,
        beta=1,
        GraphRegWeight=0.1,
    )
    # test potential gradient
    np.random.seed(1)
    log_x_hat, log_theta, mean_z, log_var_z, z_k, p_k = m.forward(x, c)
    loss = m.loss(
        x=x,
        log_x_hat=log_x_hat,
        log_theta=log_theta,
        mean_z=mean_z,
        log_var_z=log_var_z,
        z_k=z_k,
        p_k=p_k,
        obs_idx=obs_idx,
        obs_nei_idx=obs_nei_idx,
        obs_nei_conn=obs_nei_conn,
    )
    print(f"loss : {loss}")
    z_0 = m.reparameterization(mean_z, torch.exp(0.5 * log_var_z))
    print(m._dU_dz(z_0, x, c))
    p_0 = m.Temp0 * torch.randn_like(z_0)
    print(m._his(z_0, p_0, x, c))
