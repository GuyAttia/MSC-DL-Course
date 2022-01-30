import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        from .autoencoder import EncoderCNN
        conv_in_channels, _, _ = in_size
        self.cnn = EncoderCNN(conv_in_channels, 256)
        
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
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        cnn_out = self.cnn(x).view(x.shape[0],-1)
        y = self.out_layer(cnn_out)        
        # ========================
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

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        #hint (you dont have to use....)
        self.featuremap_size = featuremap_size
        self.out_channels = out_channels

        self.in_channels = 512
        self.in_layer = nn.Linear(z_dim, featuremap_size*featuremap_size*self.in_channels, bias=False)

        # from hw3.autoencoder import DecoderCNN

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
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        latent_samples = torch.randn(size=(n, self.z_dim), device=device)
        if with_grad:
            samples = self.forward(latent_samples)
        else:
            with torch.no_grad():
                samples = self.forward(latent_samples)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        in_layer_output = self.in_layer(z)
        in_layer_output = torch.reshape(in_layer_output, [z.shape[0], self.in_channels, self.featuremap_size, self.featuremap_size])    
        x = self.cnn(in_layer_output)
        # ========================
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
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    data_noise = torch.rand(y_data.shape).to(y_data.device) * label_noise - label_noise / 2
    generated_noise = torch.rand(y_generated.shape, device=y_data.device) * label_noise - label_noise / 2
    
    data_labels = data_label + data_noise
    
    generated_labels = (1-data_label) + generated_noise
    criterion = torch.nn.BCEWithLogitsLoss()
    
    loss_data = criterion(y_data, data_labels)
    loss_generated = criterion(y_generated, generated_labels)
    # ========================
    return loss_data + loss_generated


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
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, torch.full(y_generated.shape, float(data_label),device=y_generated.device))
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    gen_data = gen_model.sample(x_data.shape[0], with_grad=False)
    gen_data_score = dsc_model(gen_data)

    y_score = dsc_model(x_data)
    
    dsc_optimizer.zero_grad()
    
    dsc_loss = dsc_loss_fn(y_score, gen_data_score)
    
    dsc_loss.backward()
    
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    disc_data = dsc_model(gen_model.sample(x_data.shape[0], with_grad=True))
    
    gen_optimizer.zero_grad()
    
    gen_loss = gen_loss_fn(disc_data)
    
    gen_loss.backward()
    
    gen_optimizer.step()
    # ========================

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

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    torch.save(gen_model, checkpoint_file)
    saved = True
    # ========================

    return saved
