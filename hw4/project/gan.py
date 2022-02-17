import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.utils import spectral_norm

from typing import Callable


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        
        from hw4.autoencoder import EncoderCNN
        conv_in_channels, _, _ = in_size
        self.cnn = EncoderCNN(conv_in_channels, 256)
        
        with torch.no_grad():
            num_of_features = 1
            num_of_features_arr = self.cnn(torch.zeros(1,*in_size)).shape
            for f in num_of_features_arr:
                num_of_features *= f
        
        self.out_layer = nn.Linear(num_of_features, 1, bias=True)

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        cnn_out = self.cnn(x).view(x.shape[0],-1)
        y = self.out_layer(cnn_out)        
        return y
    
    
class WganDiscriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        self.conv = self.make_discriminator(in_size[0])
        self.eval()

    @staticmethod
    def make_discriminator(in_size):
        modules = []
        for channel_in, channel_out in [(in_size, 64), (64, 128), (128, 256), (256, 512)]:
            modules.append(nn.Conv2d(in_channels=channel_in, 
                                     out_channels=channel_out,
                                     kernel_size=4, 
                                     padding=1, 
                                     stride=2, 
                                     bias=False))
            if channel_in != in_size:
                modules.append(nn.BatchNorm2d(num_features=channel_out))
            modules.append(nn.ReLU())
            
        # modules.append(nn.Linear(512, 1, bias=True))
        modules.append(nn.Conv2d(in_channels=512, 
                                 out_channels=1,
                                 kernel_size=4, 
                                 padding=0, 
                                 stride=1, 
                                 bias=False))
        

        return nn.Sequential(*modules)
    
    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        y = self.conv(x)
        y = torch.squeeze(y, 3)
        y = torch.squeeze(y, 2)
        return y


class SNDiscriminator(nn.Module):
    def __init__(self, in_size, out_channels):
        super().__init__()

        modules = []
        conv_in_channels = in_size[0]
        channels = [64, 128, 256]
        for l in range(3):
            modules.append(spectral_norm(nn.Conv2d(in_channels=conv_in_channels, 
                                     out_channels=channels[l],
                                     kernel_size=5,
                                     stride=2,
                                     padding=2)))
            modules.append(nn.MaxPool2d(kernel_size=3,
                                       padding=1,
                                       stride=1))
            modules.append(nn.BatchNorm2d(num_features=channels[l], eps=1e-6, momentum=0.9))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout2d(p=0.3))
            conv_in_channels = channels[l]

        modules.append(spectral_norm(nn.Conv2d(in_channels=channels[-1],
                                 out_channels=out_channels,
                                 kernel_size=5,
                                 padding=2)))
        # ========================
        self.cnn = nn.Sequential(*modules)

        with torch.no_grad():
            num_of_features = 1
            num_of_features_arr = self.cnn(torch.zeros(1,*in_size)).shape
            for f in num_of_features_arr:
                num_of_features *= f
        
        self.out_layer = nn.Linear(num_of_features, 1, bias=True)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        cnn_out = self.cnn(x).view(x.shape[0],-1)
        y = self.out_layer(cnn_out)        
        return y
   
    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        y = self.conv(x)
        y = torch.squeeze(y, 3)
        y = torch.squeeze(y, 2)
        return y
    
class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        self.featuremap_size = featuremap_size
        self.out_channels = out_channels

        self.in_channels = 512
        self.in_layer = nn.Linear(z_dim, featuremap_size*featuremap_size*self.in_channels, bias=False)

        channels = [512, 256, 128, 64, out_channels]
        modules = []
        modules.append(nn.ConvTranspose2d(in_channels=self.in_channels,
                                          out_channels=channels[0],
                                          kernel_size=5,
                                          padding=2))
        
        for i in range(4):
            modules.append(nn.BatchNorm2d(num_features=channels[i], eps=1e-6, momentum=0.9))
            modules.append(nn.ReLU())
            modules.append(nn.ConvTranspose2d(in_channels=channels[i],
                                              out_channels=channels[i + 1],
                                              kernel_size=5,
                                              stride=2,
                                              padding=2,
                                              output_padding=1))

        self.cnn = nn.Sequential(*modules)

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        
        latent_samples = torch.randn(size=(n, self.z_dim), device=device)
        if with_grad:
            samples = self.forward(latent_samples)
        else:
            with torch.no_grad():
                samples = self.forward(latent_samples)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        in_layer_output = self.in_layer(z)
        in_layer_output = torch.reshape(in_layer_output, [z.shape[0], self.in_channels, self.featuremap_size, self.featuremap_size])    
        x = self.cnn(in_layer_output)
        return x

    
class WganGenerator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        h_size = 64

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, h_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(h_size * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 8, h_size * 4, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 4, h_size * 2, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 2, h_size, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size, out_channels, featuremap_size, 2, 1, bias=False),
            # nn.Tanh()
        )

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device

        samples = torch.randn(n, self.z_dim, device=device)
        if with_grad:
            samples = self.forward(samples)
        else:
            with torch.no_grad():
                samples = self.forward(samples)
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        x = self.decoder(z)
        return x



def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    data_noise = torch.rand(y_data.shape).to(y_data.device) * label_noise - label_noise / 2
    generated_noise = torch.rand(y_generated.shape, device=y_data.device) * label_noise - label_noise / 2
    
    data_labels = data_label + data_noise
    
    generated_labels = (1-data_label) + generated_noise
    criterion = torch.nn.BCEWithLogitsLoss()
    
    loss_data = criterion(y_data, data_labels)
    loss_generated = criterion(y_generated, generated_labels)
    return loss_data + loss_generated


def discriminator_loss_fn_wgan(y_data, y_generated):
    loss_data = y_data.mean(0).view(1)
    loss_generated = y_generated.mean(0).view(1)
    return loss_data, loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, torch.full(y_generated.shape, float(data_label),device=y_generated.device))
    return loss

def generator_loss_fn_wgan(y_generated):
    loss = y_generated.mean(0).view(1)
    return loss

def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
    name,
    hp,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
    gen_data = gen_model.sample(x_data.shape[0], with_grad=False)
    gen_data_score = dsc_model(gen_data)
    y_score = dsc_model(x_data)
    
    dsc_optimizer.zero_grad()
    
    if name == 'gan' or name == 'sngan':
        dsc_loss = dsc_loss_fn(y_score, gen_data_score)
        dsc_loss.backward()
        dsc_optimizer.step()
    if name == 'wgan' or name == 'snwgan':
        dsc_batch_losses = []
        for p in dsc_model.parameters():
            p.requires_grad = True
        for d_iter in range(hp['n_critic']):
            dsc_model.zero_grad()
            for p in dsc_model.parameters():
                p.data.clamp_(-hp['c'], hp['c'])
            dsc_loss_real, dsc_loss_gen = dsc_loss_fn(y_score, gen_data_score)
            dsc_loss_real.backward()
            dsc_loss_gen.backward(torch.Tensor([-1]).to(x_data.device))
            dsc_iter_loss = dsc_loss_real - dsc_loss_gen
            dsc_batch_losses.append(dsc_iter_loss.item())
            dsc_optimizer.step()
        dsc_loss = np.mean(dsc_batch_losses)
        
    disc_data = dsc_model(gen_model.sample(x_data.shape[0], with_grad=True))
    
    gen_optimizer.zero_grad()
    
    gen_loss = gen_loss_fn(disc_data)
    
    gen_loss.backward()
    
    gen_optimizer.step()

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    torch.save(gen_model, checkpoint_file)
    saved = True

    return saved




