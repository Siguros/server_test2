import torch
from torch import nn


class Encoder(nn.Module):
    """Encoder."""

    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.FC_input = nn.Linear(input_size, hidden_size)
        self.FC_input2 = nn.Linear(hidden_size, hidden_size)
        self.FC_mean = nn.Linear(hidden_size, latent_size)
        self.FC_var = nn.Linear(hidden_size, latent_size)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        """Forward method."""
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.FC_hidden = nn.Linear(latent_size, hidden_size)
        self.FC_hidden2 = nn.Linear(hidden_size, hidden_size)
        self.FC_output = nn.Linear(hidden_size, output_size)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Forward method."""
        batch_size, _ = x.size()
        x = x.view(batch_size, -1)
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class SimpleVAE(nn.Module):
    """Simple VAE."""

    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        """Reparameterization trick."""
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        """Forward method."""
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


if __name__ == "__main__":
    _ = SimpleVAE()
