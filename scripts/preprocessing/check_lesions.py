import sys
sys.path.append("..")
from utils import load_lesions
import argparse
from config import *


parser = argparse.ArgumentParser(description="Check that original and deformed lesions can be correctly loaded.")
parser.add_argument('seeds', help="Seeds of the deformed versions to check", type=int, nargs='*', metavar='N', default=[])
args = parser.parse_args()

to_compute = [None, ]
for i in to_compute + args.seeds:
    lpp = load_lesions(PATCH_SIZE, deformed=i)
    
    neg = 0
    pos = 0
    for db_id in lpp:
        for pat in lpp[db_id]:
            for les in lpp[db_id][pat]:
                if les // 2000 == 1:
                    neg += 1
                else:
                    pos += 1
    print(f"[RESULT] #Lesions: {neg}/{pos} (-/+) - {neg+pos} in total.")
