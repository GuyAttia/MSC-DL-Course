import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        conv_in_channels = in_channels
        channels = [64, 128, 256]
        for l in range(3):
            modules.append(nn.Conv2d(in_channels=conv_in_channels, 
                                     out_channels=channels[l],
                                     kernel_size=5,
                                     stride=2,
                                     padding=2))
            modules.append(nn.MaxPool2d(kernel_size=3,
                                       padding=1,
                                       stride=1))
            modules.append(nn.BatchNorm2d(num_features=channels[l], eps=1e-6, momentum=0.9))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout2d(p=0.3))
            conv_in_channels = channels[l]

        modules.append(nn.Conv2d(in_channels=channels[-1],
                                 out_channels=out_channels,
                                 kernel_size=5,
                                 padding=2))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        channels = [256, 128, 64, out_channels]
        modules.append(nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=channels[0],
                                          kernel_size=5,
                                          padding=2))
        
        for l in range(3):
            modules.append(nn.BatchNorm2d(num_features=channels[l], eps=1e-6, momentum=0.9))
            modules.append(nn.ReLU())
            modules.append(nn.ConvTranspose2d(in_channels=channels[l], 
                                     out_channels=channels[l+1],
                                     kernel_size=5,
                                     stride=2,
                                     padding=2,
                                     output_padding=1))
            modules.append(nn.Dropout2d(p=0.3))
            
            conv_in_channels = channels[l]
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.gen_mu = nn.Linear(in_features=n_features, out_features=z_dim)
        self.gen_logvar = nn.Linear(in_features=n_features, out_features=z_dim)
        self.z_to_h = nn.Linear(in_features=z_dim, out_features=n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO: 
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        # Extract image features
        h = self.features_encoder(x).reshape(x.shape[0], -1)
        # Transform into mu and log_sigma2
        mu = self.gen_mu(h)
        log_sigma2 = self.gen_logvar(h)
        # Reparametrization trick: Sample u from an isotropic gaussian and
        sig = torch.exp(log_sigma2*0.5)
        reparametrize = torch.randn(mu.size(), device=mu.device)
        # use it to create z.
        z = mu + reparametrize * sig
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.z_to_h(z).reshape(-1, *self.features_shape)
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            gaussian = torch.randn(size=(n,self.z_dim), device=device)
            samples = self.decode(gaussian)
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    x_dim = x.shape[1] * x.shape[2] * x.shape[3]
    data_loss = ((x - xr).view(x.shape[0], -1).norm(dim=1) ** 2)
    data_loss = data_loss.mean() / (x_sigma2 * x_dim)

    t = z_log_sigma2.exp().sum(dim=1)
    log_det = -z_log_sigma2.sum(dim=1)
    
    d_z = z_mu.shape[1]
    
    mu_norm = z_mu.norm(dim=1) ** 2

    kldiv_vec = t + mu_norm + log_det
    kldiv_loss = kldiv_vec.mean() - d_z

    loss = data_loss + kldiv_loss
    # print(loss)
    # ========================

    return loss, data_loss, kldiv_loss
