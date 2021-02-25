import crypten
import crypten.mpc as mpc
import torch

import random
from crypten.autograd_cryptensor import AutogradContext, AutogradCrypTensor
from crypten.common.tensor_types import is_float_tensor
from crypten.gradients import AutogradFunction

import logging
import random
import unittest

import crypten
import crypten.gradients as gradients
import torch
from crypten.autograd_cryptensor import AutogradContext, AutogradCrypTensor
from crypten.common.tensor_types import is_float_tensor
from crypten.gradients import AutogradFunction

@mpc.run_multiprocess(world_size=2)
def test_mp1d():
    #x_small = torch.rand(1, 3, 8, 8)
    # mp = torch.nn.MaxPool2d(2, return_indices=True)
    # res, ind = mp(x_small)
    # print(ind)
    # x_crypt = mpc.MPCTensor(x_small)

    #crypten.func.max_pool1d(x_crypt)

    x_small = torch.rand(10, 3, 28)

    print(x_small)
    mp = torch.nn.MaxPool1d(2, return_indices=True)
    res, ind = mp(x_small)
    input = AutogradCrypTensor(crypten.cryptensor(x_small))
    outputs = input.max_pool1d(2, return_indices=True)
    out = outputs[0].get_plain_text()
    assert torch.all(torch.eq(out, res))
    #outputs[0] = input.max_pool2d(2, return_indices=False)

crypten.init()
test_mp1d()