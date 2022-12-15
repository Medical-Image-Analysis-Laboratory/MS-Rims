import argparse
import sys
sys.path.append("..")
from utils import extract_lesions_from_source_data, generate_deformed_lesions_files
from config import *


parser = argparse.ArgumentParser(description="Generate deformed versions of the data using a specified seed.")
parser.add_argument('seed', help="The random seed to use in the generation of the deformation matrix.", type=int)
args = parser.parse_args()

generate_deformed_lesions_files(args.seed, PATCH_SIZE_DEFORMATIONS, PATCH_SIZE)