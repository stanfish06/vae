import pytorch_lightning as pl
import torch
import torch.nn as nn

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, input_e_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        combined_input_dim = input_dim + condition_dim
        self.FC_input1 = nn.Linear(combined_input_dim, hidden_dim)
        self.FC_hidden1 = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_input2 = nn.Linear(combined_input_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.batch_norm_input = nn.BatchNorm1d(input_e_dim, momentum=0.01, eps=0.001)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x_e, x_l, c):
        # x_norm = self.batch_norm_input(x)
        x_e_norm = self.batch_norm_input(x_e)
        x_c = torch.cat([x_e_norm, x_l, c], dim=1)
        _h1 = self.dropout(
            self.activation_fn(self.layer_norm(self.batch_norm(self.FC_input1(x_c))))
        )
        h1 = self.dropout(
            self.activation_fn(
                self.layer_norm(
                    self.batch_norm(self.FC_hidden1(torch.concat([_h1, c], dim=1)))
                )
            )
        )
        _h2 = self.dropout(
            self.activation_fn(self.layer_norm(self.batch_norm(self.FC_input2(x_c))))
        )
        h2 = self.dropout(
            self.activation_fn(
                self.layer_norm(
                    self.batch_norm(self.FC_hidden2(torch.concat([_h2, c], dim=1)))
                )
            )
        )
        mean = self.FC_mean(h1)
        log_var = self.FC_var(h2)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_dim, condition_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        combined_latent_dim = latent_dim + condition_dim
        self.FC_latent1 = nn.Linear(combined_latent_dim, hidden_dim)
        self.FC_hidden1 = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_latent2 = nn.Linear(combined_latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim + condition_dim, hidden_dim)
        self.FC_output_mean = nn.Linear(hidden_dim, output_dim)
        self.FC_output_var = nn.Linear(hidden_dim, output_dim)
        # self.FC_output_var = nn.Parameter(torch.zeros(output_dim))
        self.activation_fn = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        _h1 = self.dropout(
            self.activation_fn(self.layer_norm(self.batch_norm(self.FC_latent1(x_c))))
        )
        h1 = self.dropout(
            self.activation_fn(
                self.layer_norm(
                    self.batch_norm(self.FC_hidden1(torch.concat([_h1, c], dim=1)))
                )
            )
        )
        _h2 = self.dropout(
            self.activation_fn(self.layer_norm(self.batch_norm(self.FC_latent2(x_c))))
        )
        h2 = self.dropout(
            self.activation_fn(
                self.layer_norm(
                    self.batch_norm(self.FC_hidden2(torch.concat([_h2, c], dim=1)))
                )
            )
        )
        x_hat = self.FC_output_mean(h1)
        log_var_x_hat = self.FC_output_var(h2)

        return x_hat, log_var_x_hat


class VAE(pl.LightningModule):
    def __init__(
        self,
        Encoder,
        Decoder,
        x_e_dim,
        x_l_dim,
        x_l_weight=10.0,
        use_size_weighting=True,
        kl_weight=0.5,
    ):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.x_e_dim = x_e_dim
        self.x_l_dim = x_l_dim
        self.x_l_weight = x_l_weight  # Higher weight for x_l reconstruction
        self.use_size_weighting = use_size_weighting
        self.kl_weight = kl_weight
        # Calculate size-based weights
        if use_size_weighting:
            total_dim = x_e_dim + x_l_dim
            self.x_e_size_weight = total_dim / x_e_dim  # Higher for smaller components
            self.x_l_size_weight = total_dim / x_l_dim
        else:
            self.x_e_size_weight = 1.0
            self.x_l_size_weight = 1.0

    def reparameterization(self, mean, sd):
        epsilon = torch.randn_like(sd)
        z = mean + sd * epsilon
        return z

    def forward(self, x_e, x_l, c):
        mean, log_var = self.Encoder(x_e, x_l, c)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat, log_var_x_hat = self.Decoder(z, c)
        return x_hat, log_var_x_hat, mean, log_var

    def loss(self, x_e, x_l, x_hat, log_var_x_hat, mean, log_var):
        from torch.distributions import Independent, Normal
        from torch.distributions.kl import kl_divergence

        # Split reconstructed output
        x_e_hat = x_hat[:, : self.x_e_dim]
        x_l_hat = x_hat[:, self.x_e_dim :]

        # Split variance
        log_var_x_e_hat = log_var_x_hat[:, : self.x_e_dim]
        log_var_x_l_hat = log_var_x_hat[:, self.x_e_dim :]

        # Separate reconstruction losses
        sigma_x_e = torch.exp(0.5 * log_var_x_e_hat)
        sigma_x_l = torch.exp(0.5 * log_var_x_l_hat)

        pxe_z = Independent(Normal(x_e_hat, sigma_x_e), 1)
        pxl_z = Independent(Normal(x_l_hat, sigma_x_l), 1)

        reconst_x_e = -pxe_z.log_prob(x_e)
        reconst_x_l = -pxl_z.log_prob(x_l)
        # Combined weighting: size-based + semantic importance
        final_x_e_weight = self.x_e_size_weight
        final_x_l_weight = self.x_l_size_weight * self.x_l_weight

        weighted_reconst = (
            final_x_e_weight * reconst_x_e + final_x_l_weight * reconst_x_l
        )

        # sigma_x = torch.exp(0.5 * log_var_x_hat)
        # pxz = Independent(Normal(x_hat, sigma_x), 1)
        # reconst = -pxz.log_prob(torch.concat([x_e, x_l], -1))  # shape: (batch,)
        avg_reconst_weight = (final_x_e_weight + final_x_l_weight) / 2
        # KL term between q(z|x) and p(z)
        sigma_z = torch.exp(0.5 * log_var)
        qzx = Independent(Normal(mean, sigma_z), 1)
        pz = Independent(Normal(torch.zeros_like(mean), torch.ones_like(sigma_z)), 1)
        kl = kl_divergence(qzx, pz)  # shape: (batch,)

        return (weighted_reconst + avg_reconst_weight * self.kl_weight * kl).mean()

    def training_step(self, batch, batch_idx):
        x_e, x_l, c = batch
        x_e = x_e.view(x_e.size(0), -1)
        x_l = x_l.view(x_l.size(0), -1)
        x_hat, log_var_x_hat, mean, log_var = self.forward(x_e, x_l, c)
        loss = self.loss(x_e, x_l, x_hat, log_var_x_hat, mean, log_var)
        values = {"loss": loss}  # add more items if needed
        self.log_dict(values, prog_bar=True)
        return loss

    def configure_optimizers(self):  # Fixed typo from configure_optimizes
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    pass
    # # Example usage
    # input_dim = 10
    # condition_dim = 5
    # hidden_dim = 50
    # latent_dim = 5

    # # Approach 1: Simple concatenation
    # encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim)
    # decoder = Decoder(input_dim, condition_dim, hidden_dim, latent_dim)
    # model = VAE(encoder, decoder)

    # # Test
    # x = torch.randn(3, input_dim)
    # c = torch.zeros(3, condition_dim)
    # c[0, 0] = 1
    # c[1, 1] = 1
    # c[2, 2] = 1

    # x_hat, log_var_x_hat, mean, log_var = model.forward(x, c)
    # print("Approach 1 - Concatenation:")
    # print(f"x_hat shape: {x_hat.shape}")
    # print(f"Reconstruction loss: {model.loss(x, x_hat, log_var_x_hat, mean, log_var)}")
