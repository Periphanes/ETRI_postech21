import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir-result', type=str, default='.')
parser.add_argument('--project-name', type=str, default='proj')
parser.add_argument('--seed-list', type=list, default=[0,42,7,89,4])

parser.add_argument('--epochs', type=int, default=100)



args = parser.parse_args()
args.dir_root = os.getcwd()