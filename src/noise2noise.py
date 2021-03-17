#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable

from unet import UNet
from utils import *

import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()


    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        self.model = UNet(in_channels=3)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'sampling':
                self.loss = SamplingLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = "supersample"

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/supersample-model.pt'.format(self.ckpt_dir)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.8f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)


    def test(self, test_loader, show):
        self.model.train(False)

        source_imgs = []
        img_generated_list = []
        clean_imgs = []

        # Create directory for generated images
        result_dir = os.path.dirname(self.p.results)
        save_path = os.path.join(result_dir, 'DL-generated')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):

            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            img_generated = self.model(source).detach()
            img_generated_list.append(img_generated)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        img_generated_list = [t.squeeze(0) for t in img_generated_list]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.imgs[i]
            create_montage(img_name, save_path, source_imgs[i], img_generated_list[i], clean_imgs[i], show)


    def eval(self, valid_loader):

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()

        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            img_generated = self.model(source)

            # Update loss
            loss = self.loss(img_generated, target)
            loss_meter.update(loss.item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]

        return valid_loss, valid_time


    def train(self, train_loader, valid_loader):
        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        print("num_batches: " + str(num_batches))
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'train_loss': [],
                 'valid_loss': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                img_generated = self.model(source)

                loss = self.loss(img_generated, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class SamplingLoss(nn.Module):
    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(SamplingLoss, self).__init__()
        self._eps = eps

    def get_gradient(self, img):
        img = img.squeeze(0)
        ten = torch.unbind(img)
        x = ten[0].unsqueeze(0).unsqueeze(0)

        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        G_x = conv1(Variable(x)).data.view(1, x.shape[2], x.shape[3])

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
        G_y = conv2(Variable(x)).data.view(1, x.shape[2], x.shape[3])

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return G

    def forward(self, generated, target):
        # loss 1 mse loss
        mse_loss = nn.MSELoss()
        loss1 = mse_loss(generated, target)

        # loss 2 edge loss
        generated_edges = self.get_gradient(generated)
        target_edges = self.get_gradient(target)
        loss2 = mse_loss(generated_edges, target_edges)

        loss = 0.4 * loss1 + 0.6 * loss2

        return loss
