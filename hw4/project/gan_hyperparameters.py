# Copied from hw4 part2 
def gan_hyperparameters():
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.3,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,
        ),
        n_critic=1,
        c=0.0
    )
    # ========================
    return hypers

def sngan_hyperparams():
    hypers = gan_hyperparameters()
    new_hypers = dict(

    )
    hypers.update(new_hypers)
    return hypers



def wgan_hyperparams():
    hypers = gan_hyperparameters()
    new_hypers = dict(
        discriminator_optimizer=dict(
            type="RMSprop",
            lr=5e-4,
        ),
        generator_optimizer=dict(
            type="RMSprop",
            lr=5e-4,
        ),
        n_critic=5,
        c=0.01
    )
    hypers.update(new_hypers)
    return hypers

def w_sn_gan_hyperparams():
    hypers = gan_hyperparameters()
    new_hypers = dict(
        discriminator_optimizer=dict(
            type="RMSprop",
            lr=2e-4,
        ),
        generator_optimizer=dict(
            type="RMSprop",
            lr=2e-4,
        ),
        n_critic=5,
        c=0.01
    )
    hypers.update(new_hypers)
    return hypers