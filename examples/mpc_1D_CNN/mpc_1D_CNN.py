#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import crypten.communicator as comm

import datetime

import crypten
import torch

from examples.meters import AverageMeter


def train_1D_CNN(features, labels, epochs=50, lr=0.5, print_time=False):
    return


def evaluate_1D_CNN(features, labels, w, b):
    return


def run_mpc_1D_CNN(
    CNN_pth="../checkpoint.pth", data_pth = "../test/embeddings_tensor.pth", label_pth="../test/bin_labels_tensor.pth",
    count = 0, batch = False
):
    crypten.init()

    ALICE = 0
    BOB = 1

    rank = comm.get().get_rank()

    from nets import Net6
    dummy_model = Net6()

    plaintext_model = crypten.load(CNN_pth, dummy_model=dummy_model, src=ALICE)

    dummy_input = torch.empty((1, 1, 768))
    dummy_input.to('cpu')

    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt()
    private_model.eval()

    data_enc = crypten.load(data_pth, src=BOB)

    labels = None
    if rank == BOB:
        labels = torch.load(label_pth)

    if count == 0:
        count = len(data_enc)

    input = data_enc[:count]

    correctly_classified = 0
    incorrectly_classified = 0

    start = datetime.datetime.now()

    if batch:
        classification = private_model(input)
    else:
        for i in range(count):
            #print(i)
            classify = private_model(input[i])
            #classify = classify.get_plain_text()

            # if rank == BOB:
            #     classification = torch.argmax(classify)
            #     # print(classification)
            #     # print(labels[i])
            #     if classification == labels[i]:
            #         correctly_classified += 1
            #     else:
            #         incorrectly_classified += 1

    end = datetime.datetime.now()
    #print(str(correctly_classified / incorrectly_classified))
    print("count = " + str(count) + ", " + str(((end - start).total_seconds() * 1000)//1) + "ms")

