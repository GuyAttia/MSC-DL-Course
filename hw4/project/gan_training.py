# from .inception_score import inception_score
#                 if num_epochs - epoch_idx <= self.inc_score_epochs:
#                     inc_score, _ = inception_score(self.gen.sample(100, with_grad=False), resize=True)
#                     print(f'Epoch {epoch_idx + 1}, Inception score:{inc_score}')
#                     if inc_score > max_inc_score:
#                         max_inc_score = inc_score
#                         torch.save(self.gen, self.checkpoint_file)
#                         print(f'Saved checkpoint.')

import abc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import IPython.display
import tqdm

import cs3600.plot as plot
from project.gan_hyperparameters import *
import project.gan as gan

# from hw4.gan import train_batch, save_checkpoint

torch.manual_seed(42)

class Train(abc.ABC):
    def __init__(self, name='gan', dataset='', device='cpu'):
        self.name = name
        self.device = device
        # Hyperparams
        if name == 'gan':
            self.hp = gan_hyperparameters()

        #Data 
        batch_size = self.hp['batch_size']
        self.dl_train = DataLoader(dataset, batch_size, shuffle=True)
        
        # Model
        im_size = dataset[0][0].shape
        z_dim = self.hp['z_dim']
        if name == 'gan':
            self.dsc = gan.Discriminator(im_size).to(device)
            self.gen = gan.Generator(z_dim, featuremap_size=4).to(device)

        # Optimizer
        def create_optimizer(model_params, opt_params):
            opt_params = opt_params.copy()
            optimizer_type = opt_params['type']
            opt_params.pop('type')
            return optim.__dict__[optimizer_type](model_params, **opt_params)
        self.dsc_optimizer = create_optimizer(self.dsc.parameters(), self.hp['discriminator_optimizer'])
        self.gen_optimizer = create_optimizer(self.gen.parameters(), self.hp['generator_optimizer'])

        # Training
        if name == 'gan':
            self.checkpoint_file = 'project_checkpoints/gan'
        self.checkpoint_file_final = f'{self.checkpoint_file}_final'
        if os.path.isfile(f'{self.checkpoint_file}.pt'):
            os.remove(f'{self.checkpoint_file}.pt')

        # Show hypers
        print(self.hp)
        
    
    # Loss
    def dsc_loss_fn(self, y_data, y_generated):
        if self.name == 'gan':
            return gan.discriminator_loss_fn(y_data, y_generated, self.hp['data_label'], self.hp['label_noise'])

    def gen_loss_fn(self, y_generated):
        if self.name == 'gan':
            return gan.generator_loss_fn(y_generated, self.hp['data_label'])


    def train(self, num_epochs = 100):
        if os.path.isfile(f'{self.checkpoint_file_final}.pt'):
            print(f'*** Loading final checkpoint file {self.checkpoint_file_final} instead of training')
            num_epochs = 0
            gen = torch.load(f'{self.checkpoint_file_final}.pt', map_location=self.device)
            self.checkpoint_file = self.checkpoint_file_final

        try:
            dsc_avg_losses, gen_avg_losses = [], []
            for epoch_idx in range(num_epochs):
                # We'll accumulate batch losses and show an average once per epoch.
                dsc_losses, gen_losses = [], []
                print(f'--- EPOCH {epoch_idx+1}/{num_epochs} ---')

                with tqdm.tqdm(total=len(self.dl_train.batch_sampler), file=sys.stdout) as pbar:
                    for batch_idx, (x_data, _) in enumerate(self.dl_train):
                        x_data = x_data.to(self.device)
                        dsc_loss, gen_loss = gan.train_batch(
                            self.dsc, self.gen,
                            self.dsc_loss_fn, self.gen_loss_fn,
                            self.dsc_optimizer, self.gen_optimizer,
                            x_data)
                        dsc_losses.append(dsc_loss)
                        gen_losses.append(gen_loss)
                        pbar.update()

                dsc_avg_losses.append(np.mean(dsc_losses))
                gen_avg_losses.append(np.mean(gen_losses))
                print(f'Discriminator loss: {dsc_avg_losses[-1]}')
                print(f'Generator loss:     {gen_avg_losses[-1]}')

                if gan.save_checkpoint(self.gen, dsc_avg_losses, gen_avg_losses, self.checkpoint_file):
                    print(f'Saved checkpoint.')


                samples = self.gen.sample(5, with_grad=False)
                fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6,2))
                IPython.display.display(fig)
                plt.close(fig)
        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')