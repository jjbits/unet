#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_supersample_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('--load-ckpt', help='load model checkpoint', default=None)
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../saved-model')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', default=1, type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', default=1, type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=1, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=1, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'sampling'], default='sampling', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    datapath = "/mnt/data/supersample_data/"

    # Train/valid datasets
    train_loader = load_supersample_dataset(datapath + 'train/input', datapath + 'train/target', params.train_size, params, shuffled=True, single=False)
    valid_loader = load_supersample_dataset(datapath + 'valid/input', datapath + 'valid/target', params.valid_size, params, shuffled=False, single=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    if params.load_ckpt != None:
        n2n.load_model(params.load_ckpt)
    n2n.train(train_loader, valid_loader)
