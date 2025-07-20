import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self,
        Encoder,
        Decoder,
        nLeapFrog,
        InitLeapFrogStep,
        MaxLeapFrogStep,
        lr,
        beta,
        init_alpha,
        GraphRegWeight,
        MMD,
        MMDdim,
        MMDgamma,
        USE_VAMP,
        VAMP_K,
    ):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.nLeapFrog = nLeapFrog
        self.InitLeapFrogStep = InitLeapFrogStep
        self.MaxLeapFrogStep = MaxLeapFrogStep
        self.lr = lr
        self.beta = beta
        self.init_alpha = init_alpha
        self.GraphRegWeight = GraphRegWeight
        self.MMD = MMD
        self.MMDdim = MMDdim
        self.MMDgamma = MMDgamma
        self.USE_VAMP = USE_VAMP
        self.VAMP_K = VAMP_K

        init_alphas = self.init_alpha * np.ones(self.nLeapFrog)
        init_alphas_reparam = np.log(init_alphas / (1 - init_alphas))
        self.alphas_reparam = nn.Parameter(
            torch.from_numpy(init_alphas_reparam.astype(np.float32))
        )

        # learnable leapfrog step size
        init_lf = self.InitLeapFrogStep * np.ones(self.Encoder.latent_dim)
        init_lf_reparam = np.log(
            init_lf / (self.MaxLeapFrogStep - self.InitLeapFrogStep)
        )
        init_lf_reparam = np.tile(init_lf_reparam, (self.nLeapFrog, 1))

        self.lf_reparam = nn.Parameter(
            torch.from_numpy(init_lf_reparam.astype(np.float32))
        )

        # VAMP prior
        # Remember to exp when using it
        if USE_VAMP:
            self.log_pseudo_input = nn.Parameter(
                torch.randn(
                    self.Encoder.condition_dim, self.VAMP_K, self.Encoder.input_dim
                )
            )

    def _his(self, z_0, p_0, x, c, lf_eps):
        _z = z_0
        _p = p_0
        for k in range(self.nLeapFrog):
            p_half = _p - 1.0 / 2 * lf_eps[k, :] * self._dU_dz(_z, x, c)
            _z = _z + lf_eps[k, :] * p_half
            _p = p_half - 1.0 / 2 * lf_eps[k, :] * self._dU_dz(_z, x, c)
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
        # Calculate alphas and T_0
        alphas = torch.sigmoid(self.alphas_reparam)
        T_0 = torch.prod(alphas) ** (-2)

        # Calculate leapfrog steps
        lf_eps = self.MaxLeapFrogStep * torch.sigmoid(self.lf_reparam)

        mean_z, log_var_z = self.Encoder(x, c)
        z_0 = self.reparameterization(mean_z, torch.exp(0.5 * log_var_z))
        p_0 = torch.sqrt(T_0) * torch.randn_like(z_0)
        (z_k, p_k) = self._his(z_0, p_0, x, c, lf_eps)
        log_x_hat, log_theta = self.Decoder(z_k, c)
        return (log_x_hat, log_theta, mean_z, log_var_z, z_0, p_0, z_k, p_k)

    def get_latent_representation(self, x, c, K=10):
        z_samples = []
        # Calculate alphas and T_0
        alphas = torch.sigmoid(self.alphas_reparam)
        T_0 = torch.prod(alphas) ** (-2)
        # Calculate leapfrog steps
        lf_eps = self.MaxLeapFrogStep * torch.sigmoid(self.lf_reparam)
        mean_z, log_var_z = self.Encoder(x, c)
        for _ in range(K):
            z_0 = self.reparameterization(mean_z, torch.exp(0.5 * log_var_z))
            p_0 = torch.sqrt(T_0) * torch.randn_like(z_0)
            z_k, _ = self._his(z_0, p_0, x, c, lf_eps)
            z_samples.append(z_k.unsqueeze(0))
        z_stack = torch.cat(z_samples, dim=0)
        z_median = z_stack.median(dim=0).values
        return z_median

    # ELBO
    def loss(
        self,
        x,
        c,
        log_x_hat,
        log_theta,
        mean_z,
        log_var_z,
        z_0,
        p_0,
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

        if self.USE_VAMP:
            mean_z_prior = []
            log_var_z_prior = []
            for ci in range(self.Encoder.condition_dim):
                pseudo_x = torch.exp(self.log_pseudo_input[ci])
                pseudo_c = (
                    F.one_hot(
                        torch.tensor([ci] * self.VAMP_K),
                        num_classes=self.Encoder.condition_dim,
                    )
                    .float()
                    .to(c.device)
                )
                mean_z_ci, log_var_z_ci = self.Encoder(pseudo_x, pseudo_c)
                mean_z_prior.append(mean_z_ci)
                log_var_z_prior.append(log_var_z_ci)
            mean_z_prior = torch.cat(mean_z_prior)
            log_var_z_prior = torch.cat(log_var_z_prior)
            z = z_0.unsqueeze(1)
            mean_prior = mean_z_prior.unsqueeze(0)
            log_var_prior = log_var_z_prior.unsqueeze(0)
            var_prior = torch.clamp(torch.exp(log_var_prior), min=1e-8, max=1e8)
            log_probs = -0.5 * (
                torch.sum((z - mean_prior) ** 2 / var_prior, dim=-1)  # Mahalanobis term
                + torch.sum(log_var_prior, dim=-1)  # log |Σ|
                + z_0.shape[1] * torch.log(torch.tensor(2 * torch.pi, device=z.device))
            )
            log_p_z = torch.logsumexp(
                log_probs
                - torch.log(
                    torch.tensor(
                        self.Encoder.condition_dim * self.VAMP_K, device=z.device
                    )
                ),
                dim=1,
            )
            var_post = torch.clamp(torch.exp(log_var_z), min=1e-8, max=1e8)
            log_q_z = -0.5 * (
                torch.sum((z_0 - mean_z) ** 2 / var_post, dim=-1)
                + torch.sum(log_var_z, dim=-1)
                + z.shape[1] * torch.log(torch.tensor(2 * torch.pi, device=z.device))
            )
            # negative KL (single point estimate)
            NKL = log_p_z - log_q_z
        else:
            # NKL term between q(z|x) and p(z)
            latent_entropy_posterior = torch.sum(0.5 * log_var_z, dim=-1)
            NLL_z_prior = -0.5 * torch.sum(z_k * z_k, dim=-1)
            NLL_p_prior = -0.5 * torch.sum(p_k * p_k, dim=-1)
            NKL = (
                latent_entropy_posterior
                + NLL_z_prior
                + NLL_p_prior
                + self.Encoder.latent_dim
            )

        # graph regularization
        gLoss = torch.tensor(0.0, device=z_k.device, dtype=z_k.dtype)
        if self.GraphRegWeight > 0:
            # Create a mapping from global dataset index to local batch index
            global_to_local_idx = {
                int(global_idx): local_idx
                for local_idx, global_idx in enumerate(obs_idx)
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

        # MMD regularization
        MMDLoss = torch.tensor(0.0, device=z_k.device, dtype=z_k.dtype)
        if self.MMD > 0:
            W = torch.randn(
                z_k.size(1), self.MMDdim, device=x.device
            )  # [latent_dim, D]
            b = torch.rand(1, self.MMDdim, device=x.device) * 2 * torch.pi  # [1, D]
            scale = torch.sqrt(torch.tensor(2.0 / self.MMDgamma, device=x.device))
            norm_factor = torch.sqrt(torch.tensor(2.0 / self.MMDdim, device=x.device))

            phi_all = norm_factor * torch.cos(scale * (z_k @ W) + b)  # [batch_size, D]
            phi_mean = phi_all.mean(dim=0)  # [D]

            for k in range(c.size(1)):
                mask = c[:, k].bool()
                if mask.sum() > 1:
                    phi_k = phi_all[mask]  # [n_k, D]
                    phi_k_mean = phi_k.mean(dim=0)  # [D]
                    MMDLoss += self.MMD * torch.sum((phi_k_mean - phi_mean) ** 2)

        return (NLL_x - self.beta * NKL).mean() + gLoss + MMDLoss

    def training_step(self, batch, batch_idx):
        x, c, obs_idx, obs_nei_idx, obs_nei_conn = batch
        log_x_hat, log_theta, mean_z, log_var_z, z_0, p_0, z_k, p_k = self.forward(x, c)
        loss = self.loss(
            x=x,
            c=c,
            log_x_hat=log_x_hat,
            log_theta=log_theta,
            mean_z=mean_z,
            log_var_z=log_var_z,
            z_0=z_0,
            p_0=p_0,
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
        obs_nei_conn[(i, i + 1)] = 1.0
        obs_nei_conn[(i + 1, i)] = 1.0

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
        InitLeapFrogStep=0.01,
        MaxLeapFrogStep=0.1,
        lr=1e-4,
        beta=1,
        init_alpha=0.9,
        GraphRegWeight=0.1,
        MMD=0.1,
        MMDdim=50,
        MMDgamma=1.0,
        USE_VAMP=True,
        VAMP_K=5,
    )
    # test potential gradient
    np.random.seed(1)
    log_x_hat, log_theta, mean_z, log_var_z, z_0, p_0, z_k, p_k = m.forward(x, c)
    loss = m.loss(
        x=x,
        c=c,
        log_x_hat=log_x_hat,
        log_theta=log_theta,
        mean_z=mean_z,
        log_var_z=log_var_z,
        z_0=z_0,
        p_0=p_0,
        z_k=z_k,
        p_k=p_k,
        obs_idx=obs_idx,
        obs_nei_idx=obs_nei_idx,
        obs_nei_conn=obs_nei_conn,
    )
    print(f"loss : {loss}")
    print(f"dU_dz: {m._dU_dz(z_0, x, c).shape}")
    print(f"Latent representation: {m.get_latent_representation(x, c, K=10).shape}")
