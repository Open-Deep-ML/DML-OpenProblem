import torch
import torch.nn as nn
import torch.optim as optim

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    torch.manual_seed(seed)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    G = Generator()
    D = Discriminator()

    # Use SGD as requested
    opt_G = optim.SGD(G.parameters(), lr=learning_rate)
    opt_D = optim.SGD(D.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        # Real and fake batches
        real_data = torch.normal(mean_real, std_real, size=(batch_size, 1))
        noise = torch.randn(batch_size, latent_dim)
        fake_data = G(noise)

        # ----- Discriminator step -----
        opt_D.zero_grad()
        pred_real = D(real_data)
        pred_fake = D(fake_data.detach())
        loss_real = criterion(pred_real, torch.ones_like(pred_real))
        loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # ----- Generator step -----
        opt_G.zero_grad()
        pred_fake = D(fake_data)
        # non-saturating generator loss: maximize log D(G(z)) -> minimize -log D(G(z))
        loss_G = criterion(pred_fake, torch.ones_like(pred_fake))
        loss_G.backward()
        opt_G.step()

    return G.forward
