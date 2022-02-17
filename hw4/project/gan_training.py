import abc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import IPython.display
import tqdm
import cv2

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import cs3600.plot as plot
from project.gan_hyperparameters import *
import project.gan as gan

from .fid import calculate_fid_given_paths

torch.manual_seed(42)

class Train(abc.ABC):
    def __init__(self, name='gan', dataset='', device='cpu', load_generated_data=False):
        self.name = name
        self.device = device
        self.load_generated_data = load_generated_data
        self.fid_score_epochs = 100
        
        # Hyperparams
        if name == 'gan':
            self.hp = gan_hyperparameters()
        elif name == 'sngan':
            self.hp = sngan_hyperparameters()
        elif name == 'wgan':
            self.hp = wgan_hyperparameters()
        elif name == 'snwgan':
            self.hp = snwgan_hyperparameters()
            
        #Data 
        self.batch_size = self.hp['batch_size']
        self.dl_train = DataLoader(dataset, self.batch_size, shuffle=True)
        
        # Model
        im_size = dataset[0][0].shape
        z_dim = self.hp['z_dim']
        if name == 'gan':
            self.dsc = gan.Discriminator(im_size).to(device)
            self.gen = gan.Generator(z_dim, featuremap_size=4).to(device)
        if name == 'wgan':
            self.dsc = gan.WganDiscriminator(im_size).to(device)
            self.gen = gan.WganGenerator(z_dim, featuremap_size=4).to(device)
        if name == 'sngan':
            self.dsc = gan.SNDiscriminator(im_size, 256).to(device)
            self.gen = gan.Generator(z_dim, featuremap_size=4).to(device)
        if name == 'snwgan':
            self.dsc = gan.SNDiscriminator(im_size, 256).to(device)
            self.gen = gan.WganGenerator(z_dim, featuremap_size=4).to(device)

        # Optimizer
        def create_optimizer(model_params, opt_params):
            opt_params = opt_params.copy()
            optimizer_type = opt_params['type']
            opt_params.pop('type')
            return optim.__dict__[optimizer_type](model_params, **opt_params)
        
        self.dsc_optimizer = create_optimizer(self.dsc.parameters(), self.hp['discriminator_optimizer'])
        self.gen_optimizer = create_optimizer(self.gen.parameters(), self.hp['generator_optimizer'])

        # Training
        self.checkpoint_file = f'project_checkpoints/{name}'
        self.checkpoint_file_final = f'{self.checkpoint_file}_final'
        if os.path.isfile(f'{self.checkpoint_file}.pt'):
            os.remove(f'{self.checkpoint_file}.pt')

        # Show hypers
        print(self.hp)
        
    
    # Loss
    def dsc_loss_fn(self, y_data, y_generated):
        if self.name == 'gan' or self.name == 'sngan':
            return gan.discriminator_loss_fn(y_data, y_generated, self.hp['data_label'], self.hp['label_noise'])
        if self.name == 'wgan' or self.name == 'snwgan':
            return gan.discriminator_loss_fn_wgan(y_data, y_generated)

    def gen_loss_fn(self, y_generated):
        if self.name == 'gan' or self.name == 'sngan':
            return gan.generator_loss_fn(y_generated, self.hp['data_label'])
        if self.name == 'wgan' or self.name == 'snwgan':
            return gan.generator_loss_fn_wgan(y_generated)


    def train(self, num_epochs = 100):
        if os.path.isfile(f'{self.checkpoint_file_final}.pt'):
            print(f'*** Loading final checkpoint file {self.checkpoint_file_final} instead of training')
            num_epochs = 0
            gen = torch.load(f'{self.checkpoint_file_final}.pt', map_location=self.device)
            self.checkpoint_file = self.checkpoint_file_final

        try:
            dsc_avg_losses, gen_avg_losses = [], []
            min_fid_score = 10000000
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
                            x_data, self.name, self.hp)
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
                
                if num_epochs - epoch_idx <= self.fid_score_epochs:
                    subset, _ = next(iter(self.dl_train))
                    fid_score = calculate_fid_given_paths([subset, self.gen.sample(8, with_grad=False)], 8,self.device, 64)
                    print(f'Epoch {epoch_idx + 1}, FID score:{fid_score}')
                    if fid_score < min_fid_score:
                        min_fid_score = fid_score
                        torch.save(self.gen, self.checkpoint_file)
                        print(f'Saved checkpoint.')
                    with open(f"project/generated_data/fid_score_{self.name}.txt", 'w') as f:
                        f.write(f"{min_fid_score}")
                print(f"Finished training! best fid score is: {min_fid_score}")

            
            # Plot images from best or last model
            if os.path.isfile(f'{self.checkpoint_file_final}.pt'):
                gen = torch.load(f'{self.checkpoint_file_final}.pt', map_location=self.device)
                with open(f"project/generated_data/fid_score_{self.name}.txt", 'r') as f:
                    r = f .readlines()
                    print(f"Best fid score is: {r[0]}\n")
            print('*** Images Generated from best model:')
            if self.load_generated_data:
                IPython.display.display(IPython.display.Image(filename=f"project/generated_data/{self.name}_output.jpg"))
            else:
                samples = gen.sample(n=15, with_grad=False).cpu()
                save_image(samples, f"project/generated_data/{self.name}_image.jpg")
                fig, _ = plot.tensors_as_images(samples, nrows=3, figsize=(6,6))
                        
        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')