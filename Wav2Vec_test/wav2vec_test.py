from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import soundfile as sf
import torch
from jiwer import wer

import torchaudio
import numpy as np

from tqdm import tqdm
import os
import pickle

import os

if os.path.exists(os.path.join(os.getcwd(), 'wav2vec_processor.pickle')):
    with open('wav2vec_processor.pickle', 'rb') as file:
        processor = pickle.load(file)
else:
    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    with open('wav2vec_processor.pickle', 'wb') as file:
        pickle.dump(processor, file, pickle.HIGHEST_PROTOCOL)


target_sampling_rate = processor.feature_extractor.sampling_rate

print(f"The Target Sampling Rate: {target_sampling_rate}")

iterations = 0
file_dir = os.listdir(os.path.join(os.getcwd(), 'dataset/processed'))

speech_list = []

for data_file in file_dir:
    iterations += 1
    if iterations > 200:
        break

    with open(os.path.join('dataset/processed', data_file), 'rb') as f:
        data_point = pickle.load(f)
    
    speech_array, sampling_rate = torchaudio.load(data_point['wav_dir'])
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()

    speech_list.append(speech)

result = processor(speech_list, sampling_rate=target_sampling_rate)