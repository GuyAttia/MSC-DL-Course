r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""



# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4, h_dim=128, z_dim=256, x_sigma2=0.1, learn_rate=0.0005, betas=(0.9, 0.999),)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The $\sigma^2$ controls the variance of the normal distribution of the latent space.
By increasing it's value, we will get a wider distribution what will cause the sampled instances by the VAE to vary fromeach other and from the dataset.
By decreaing it's value, we will get a narrower distribution which is closer to the training data. 
Therefore, this parameter should be used smartly to avoid to much variance from the training set on one hand, but to avoid overfitting on the other hand.
"""

part2_q2 = r"""
**Your answer:**
1.
The reconstruction loss- it's purpose is to cause the generated data as close to the input data. It does that under the L2 norm, by minimizing it's value.
The KL loss- It's purpose is to try and make the encoder aproximate the posterior distribution correctly. This is in order to generate data closer to real data.

2.

3.
The benifit of using KL loss is to control the variance of the latent space distribution. The need to control the variance was explained in q1 above.
"""

part2_q3 = r"""
**Your answer:**
Basically we don't know the evidence distribuiton which is used to generate new data points.
In order to get close to it, we try to estimate by maximizing it and trying to get it close as possible to the real distribution.
"""

part2_q4 = r"""
**Your answer:**
We model the log of the latent space variace instead of directly model the variance itself for numeric stability.
Usually, the variance values are positive close to zero numbers. We use the log of their values to spread them across wider region of numbers (log can get values from -infinity to infinity).
Also, the log function is differentiable all over it's range, so calculating it's derivitive during backprop is easy.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.3,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,

            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
While training the GAN model, we basically optimizing the loss functions of the generator and the discriminator.
The generator is trained by generating a data point, send it to the discriminator and updating it based on the loss function. In this case, we do maintain gradients because we want to update the weights of the generator during backprop.
The discriminator it trained by getting the data from the generator, classify it, and update the loss function based on the classification. In this case, we don't update the generator wights so we don't maintain the gradients.
"""

part3_q2 = r"""
**Your answer:**
1. No, the GAN performance depends both on the loss value of the generatot and the discriminator. 
Maybe the deiscriminator is easily fooled by the generator, so the generator loss in this case will decrease although it's not generating good data.

2. It means that the generator is still fooling the discriminator, without the econd one identifing it.
In other words, the discriminator is failing on seperating between real data and fake data from the generator.
This means that the model is failing in it's pupose to train. 
"""

part3_q3 = r"""
**Your answer:**
We can see that the GAN produced more realistic and sharper images comparing to the VAE.
The VAE tries to generate images based on the MSE loss comparing the input data and the generated data. This causes the bluriness of the output. It also causes the generated images be similar to each other.
On the pther hand, the GAN's generator is not aware at all to the input data. All it has is the discriminator decision wheter it has done a good job in generating the images on not. By that, well designed generator and discriminator will produce better images.
"""

# ==============
