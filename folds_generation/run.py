import argparse
import sys
import random
import numpy as np
import os


parser = argparse.ArgumentParser(description="Generate deformed versions of the data using a specified seed.")
parser.add_argument('dataset', help="The dataset to use: 0 for BASEL, 1 for CHUV.", type=int)
parser.add_argument('seed', nargs='?', help="The random seed to use in the generation of the deformation matrix.", type=int, default=-1)
args = parser.parse_args()


from splits_generator import run

seed = args.seed
if seed == -1:
    seed = int(random.random() * 1000000)

np.random.seed(seed)
random.seed(seed)
run(args.dataset, seed)