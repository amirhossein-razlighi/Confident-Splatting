import torch
import torch.nn as nn
import torch.nn.functional as F

class SplatEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SplatEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class SplatDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(SplatDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        reconstruction = self.fc_out(h)
        return reconstruction

class SplatVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(SplatVAE, self).__init__()
        self.encoder = SplatEncoder(input_dim, latent_dim)
        self.decoder = SplatDecoder(latent_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss: mean squared error (or L1 loss)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# Example usage:
if __name__ == '__main__':
    # Assume splats are flattened into a vector of size input_dim.
    input_dim = 100  # adjust according to your splat parameters
    latent_dim = 20  # tunable latent space size
    output_dim = input_dim

    vae = SplatVAE(input_dim, latent_dim, output_dim)
    
    # Dummy input representing a batch of splats
    dummy_input = torch.randn(16, input_dim)
    recon, mu, logvar = vae(dummy_input)
    
    loss = loss_function(recon, dummy_input, mu, logvar)
    loss.backward()
    print(f"Loss: {loss.item()}")