#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_supersample_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='U-Net Supersampler.')

    # test parameters
    parser.add_argument('--results', help='test result path', default="../test_result/")
    parser.add_argument('--data', help='data path', default="../one_test_28/")
    parser.add_argument('--load-ckpt', help='load model checkpoint', default="../saved-model/supersample-5000/n2n-epoch1-0.00000637.pt")
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=1, type=int)
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true', default=False)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=0, type=int)
    parser.add_argument('--test-size', help='size of test dataset', default=100, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.redux = False
    params.clean_targets = True
    test_loader = load_supersample_dataset(params.data + 'input', params.data + 'target', params.test_size, params, shuffled=False, single=False)
    n2n.load_model(params.load_ckpt)
    n2n.test(test_loader, show=params.show_output)
