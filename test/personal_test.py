import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
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

def check(self, encrypted_tensor, reference, msg, dst=None, tolerance=None):
        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text(dst=dst)
        if dst is not None and dst != self.rank:
            self.assertIsNone(tensor)
            return

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)

        self.assertTrue(is_float_tensor(reference), "reference must be a float")
        diff = (tensor - reference).abs_()
        norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
        test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
        test_passed = test_passed.gt(0).all().item() == 1
        if not test_passed:
            logging.info(msg)
            logging.info("Result = %s;\nreference = %s" % (tensor, reference))
        self.assertTrue(test_passed, msg=msg)

@mpc.run_multiprocess(world_size=2)
def test_mp1d():
    #x_small = torch.rand(1, 3, 8, 8)
    # mp = torch.nn.MaxPool2d(2, return_indices=True)
    # res, ind = mp(x_small)
    # print(ind)
    # x_crypt = mpc.MPCTensor(x_small)

    #crypten.func.max_pool1d(x_crypt)

    x_small = torch.rand(2, 5, 10)

    #print(x_small)
    mp = torch.nn.MaxPool1d(2, return_indices=True)
    res, ind = mp(x_small)
    input = AutogradCrypTensor(crypten.cryptensor(x_small))
    outputs = input.max_pool1d(2, return_indices=True)
    out = outputs[0].get_plain_text()
    if comm.get().get_rank() == 0:
        print('----------------------------------')
        print(out)
        print(type(out))
        print('----------------------------------')
        print(res)
        print(type(res))
        assert(torch.allclose(out, res))
        check(None, outputs[0], res, msg="panic")
    #outputs[0] = input.max_pool2d(2, return_indices=False)

crypten.init()
test_mp1d()