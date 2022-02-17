# Copied from hw4 part2 
def gan_hyperparameters():
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.3,
        discriminator_optimizer=dict(
            type="Adam",
            lr=0.0005,
        ),
        generator_optimizer=dict(
            type="Adam",
            lr=0.0005,
        ),
        n_critic=1,
        c=0.0
    )
    return hypers

def sngan_hyperparameters():
    hypers = gan_hyperparameters()
    return hypers

def wgan_hyperparameters():
    hypers = gan_hyperparameters()
    new_hypers = dict(
        discriminator_optimizer=dict(
            type="RMSprop", # Using RMSprop based on WGAN paper recommendation
            lr=0.0005,
        ),
        generator_optimizer=dict(
            type="RMSprop", # Using RMSprop based on WGAN paper recommendation
            lr=0.0005,
        ),
        n_critic=1,
        c=0.01
    )
    hypers.update(new_hypers)
    return hypers

def snwgan_hyperparameters():
    hypers = gan_hyperparameters()
    new_hypers = dict(
        discriminator_optimizer=dict(
            type="RMSprop", # Using RMSprop based on WGAN paper recommendation
            lr=0.0002,
        ),
        generator_optimizer=dict(
            type="RMSprop", # Using RMSprop based on WGAN paper recommendation
            lr=0.0002,
        ),
        n_critic=1,
        c=0.01
    )
    hypers.update(new_hypers)
    return hypers