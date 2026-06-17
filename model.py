import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class VAEHyperparams:
    latent_dim: int
    encoder_kernel_shape: tuple[int] = (3,3)
    decoder_kernel_shape: tuple[int] = (4,4)
    use_softplus_std: bool = True;
    pixel_reconstruction_weight: float = 1.0;
    spectral_reconstruction_weight: float = 0.5;
    edge_reconstruction_weight:float = 1000.0
    kl_elementwise_threshold: float = 0.5;

@dataclass
class VAEOutput:
    z: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor
    reconstruction: torch.Tensor

    # total loss
    loss: torch.Tensor

    # kl and reconstruction contributions
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor

    # all reconstruction losses
    loss_recon_pixel: torch.Tensor
    loss_recon_spectral: torch.Tensor
    loss_recon_edge: torch.Tensor


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x): return self.block(x)

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Starts at 0 so the network learns to use it slowly

    def forward(self, x):
        batch, ch, h, w = x.size()
        # Flatten spatial dimensions
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)
        
        # Attention map
        attn = torch.bmm(q, k)
        attn = torch.softmax(attn, dim=-1)
        
        # Complementary attention output
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, ch, h, w)
        return x + self.gamma * out
    
class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, output_padding=0):
        super().__init__()
        # upsample using inverse convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, 
            padding=padding, output_padding=output_padding
        )
        
        # refine using traditional convolution
        self.conv_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # upsample
        x_upsampled = self.upsample(x)
        # combine refined upsample and original upsample wih skip connection
        out = x_upsampled + self.conv_refine(x_upsampled)
        return self.relu(out)

class VAE(nn.Module):
    EPSILON = 1e-8;
    def __init__(self, config: VAEHyperparams):
        super().__init__()

        self.config = config;


        # encoder
        self.encoder = nn.Sequential(

            EncoderBlock(1,8,kernel_size=config.encoder_kernel_shape),
            EncoderBlock(8,16,kernel_size=config.encoder_kernel_shape),
            EncoderBlock(16,32,kernel_size=config.encoder_kernel_shape),

            nn.Flatten(),
            nn.Linear(in_features=13824, out_features=config.latent_dim*2),

            nn.Tanh()
        )

        # linear layer to expand embedding vector prior to 2d upsampling
        self.decoder_linear = nn.Linear(config.latent_dim, 256 * 8 * 14)

        # spacial decoder
        self.decoder_conv = nn.Sequential(

            SelfAttention2d(256), 
            
            ResidualDecoderBlock(256, 128, padding=1, output_padding=(0, 1)),
            ResidualDecoderBlock(128, 64),
            ResidualDecoderBlock(64, 32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, std) -> torch.Tensor:
        """Reparameterize to sample differentiably"""
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def encode(self, x):
        """encode spectrogram"""
        encoder_output = self.encoder(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=-1)
        return mu, sigma
    def decode(self, x):
        """decode spectrogram"""
        z_expanded = self.decoder_linear(x);
        z_spacial = z_expanded.view(-1, 256, 8, 14)
        output = self.decoder_conv(z_spacial);

        # cut output down to size
        output = output[:, :, :, 4:220];
        return output;
    
    def forward(self, x, kl_weight=1.0):
        """ encode, decode and compute loss"""
        mu, sigma =  self.encode(x);
        std = self._sigma_to_std(sigma);

        z = self.reparameterize(mu, std);
    
        reconstruction = self.decode(z);

        # basic pixel MSE loss
        recon_loss_pixel = F.mse_loss(reconstruction, x, reduction="mean")
        # alternatively binary cross entropy to force reconstruction into more defined bird songs
        # recon_loss_pixel = F.binary_cross_entropy(reconstruction, x, reduction='sum')

        # spectral loss
        recon_loss_spectral = spectral_loss(reconstruction, x);

        # gradient "edge" loss 
        recon_loss_edge = edge_loss(reconstruction, x);

        # weighted sum of reconstruction loss functions determined by hyperparameters
        recon_loss = (
            self.config.pixel_reconstruction_weight*recon_loss_pixel +
            self.config.spectral_reconstruction_weight * recon_loss_spectral +
            self.config.edge_reconstruction_weight*recon_loss_edge
        )

        # KL loss that allows element wise deviations from the distribution below a certain threshold
        kl_elementwise = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1)
        kl_loss = (torch.clamp(kl_elementwise, min=self.config.kl_elementwise_threshold) - self.config.kl_elementwise_threshold).mean();
        # Alternatively traditional kl loss without thresholding
        # kl_loss = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1).mean()

        # weighted sum of reconstruction loss and KL divergence
        loss = (recon_loss + kl_loss*kl_weight);

        return VAEOutput(z, mu, std, reconstruction, loss, recon_loss, kl_loss, recon_loss_pixel, recon_loss_spectral, recon_loss_edge)

    

    def _sigma_to_std(self, sigma: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
        """Convert sigma parameter to standard deviation."""
        if self.config.use_softplus_std:
            return F.softplus(sigma) + eps # softplus
        else:
            return torch.exp(0.5 * sigma)  # sigma represents log-variance

def spectral_loss(recon, target):
    # Frobenius norm of the difference divided by Frobenius norm of the target
    return torch.norm(target - recon, p='fro') / torch.norm(target, p='fro')

def edge_loss(recon, target):
    # compares the gradients of the spectrograms rather than the pixel values in order to encourage sharp edges
    recon_grad_h = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    target_grad_h = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    recon_grad_w = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    target_grad_w = target[:, :, :, 1:] - target[:, :, :, :-1]
    
    return nn.functional.mse_loss(recon_grad_h, target_grad_h) + nn.functional.mse_loss(recon_grad_w, target_grad_w)