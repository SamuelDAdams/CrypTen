#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run mpc_linear_svm example in multiprocess mode:

$ python3 examples/mpc_linear_svm/launcher.py --multiprocess

To run mpc_linear_svm example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_linear_svm/mpc_linear_svm.py \
      examples/mpc_linear_svm/launcher.py
"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="CrypTen Linear SVM Training")
parser.add_argument(
    "--world_size",
    type=str,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "--CNN_path",
    type=str,
    default="../train/checkpoint.pth",
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "--label_path",
    type=str,
    default="../test/bin_labels_tensor.pth",
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "--data_path",
    type=str,
    default="../test/embedings_tensor.pth",
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
)

parser.add_argument(
    "--batch",
    type=int,
    default=0,
    help="Run example in multiprocess mode",
)

parser.add_argument(
    "--count",
    type=int,
    default=0,
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from mpc_1D_CNN import run_mpc_1D_CNN

    run_mpc_1D_CNN(
        args.CNN_path, args.data_path, args.label_path, args.count, args.batch
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
