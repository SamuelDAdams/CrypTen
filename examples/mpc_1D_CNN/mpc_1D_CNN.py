#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import crypten.communicator as comms

import crypten
import torch

from examples.meters import AverageMeter


def train_1D_CNN(features, labels, epochs=50, lr=0.5, print_time=False):
    return



def evaluate_1D_CNN(features, labels, w, b):
    return


def run_mpc_1D_CNN(
    CNN_pth="../train/checkpoint.pth", data_pth = "../test/embedings_tensor.pth"
):
    crypten.init()

    ALICE = 0
    BOB = 1

    from nets import Net6
    dummy_model = nets.Net6()

    plaintext_model = crypten.load(CNN_pth, dummy_model=dummy_model, src=ALICE)

    dummy_input = torch.empty((1, 1, 768))
    dummy_input.to('cpu')

    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt()
    private_model.eval()

    data_enc = crypten.load(data_pth, src=BOB)
    input = data_enc[:count]

    classification = private_model(input[0])
    print(classification.get_plain_text())
    print('done')
