import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir-result', type=str, default='.')
parser.add_argument('--project-name', type=str, default='proj')
parser.add_argument('--seed-list', type=list, default=[5, 23, 7, 89, 4])
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--datasets', type=str, default="default", choices=["default"])
parser.add_argument('--small-dataset', type=bool, default=False)

parser.add_argument('--input-types', type=str, default="static", choices=["static", "txt", "audio", "sig", "audio_txt", "audio_txt_shortform", "txt_shortform", "audio_shortform"])
parser.add_argument('--model', type=str, default="default_model")

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)

parser.add_argument('--dataset-loc', type=str, default="dataset/processed")
parser.add_argument('--result-loc', type=str, default="results")

parser.add_argument('--lr-init', type=float, default=0.001)

parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--use-return-dict', type=bool, default=False)
parser.add_argument('--final-dropout', type=float, default=0.1)
parser.add_argument('--num_labels', type=int, default=7)
parser.add_argument('--problem-type', type=str, default="multi_label_classification")
parser.add_argument('--pooling-mode', type=str, default="mean")

parser.add_argument('--audio-max-length', type=int, default=100000)
parser.add_argument('--bottleneck-length', type=int, default=16)
parser.add_argument('--audio-weight', type=float, default=0.09)
parser.add_argument('--txt-weight', type=float, default=1.0)
parser.add_argument('--transformer-layers', type=int, default=16)
parser.add_argument('--transformer-heads', type=int, default=16)
parser.add_argument('--transformer-ff-dim', type=int, default=1024)
parser.add_argument('--transformer-dropout', type=float, default=0.12)
parser.add_argument('--transformer-activation', type=str, default='gelu')

args = parser.parse_args()
args.dir_root = os.getcwd()
