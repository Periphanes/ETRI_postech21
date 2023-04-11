from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, EvalPrediction
from transformers import AutoConfig, TrainingArguments, Trainer, is_apex_available
from packaging import version
from datasets import load_dataset
from dataclasses import dataclass


import torch
import numpy as np

import transformers

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

import soundfile as sf
import torch
from jiwer import wer

import torchaudio
import numpy as np
import sys

from tqdm import tqdm
import pickle

import os
sys.path.append(os.getcwd())

from models.wav2vec2 import *
from control.config import args


config = AutoConfig.from_pretrained(
    "kresnik/wav2vec2-large-xlsr-korean",
    num_labels = args.num_labels,
    finetuning_task = "wav2vec2_clf"
)
setattr(config, 'pooling_mode', args.pooling_mode)

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
labels_list = []

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
    labels_list.append(data_point["total_emot"])

speech_split_count = len(speech_list) // 4

test_speech = speech_list[:speech_split_count]
train_speech = speech_list[speech_split_count:]

test_labels = labels_list[:speech_split_count]
train_labels = labels_list[speech_split_count:]

train_dataset = processor.__call__(audio=train_speech, sampling_rate=target_sampling_rate)
train_dataset["labels"] = train_labels

eval_dataset = processor.__call__(audio=test_speech, sampling_rate=target_sampling_rate)
eval_dataset["labels"] = test_labels

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.input_values = dataset['input_values']
        self.labels = dataset['labels']
        self.attention_mask = dataset['attention_mask']
    
    def inputs(self):
        return self.input_values
    
    def __getitem__(self, idx):
        item = {}
        item["input_values"] = torch.tensor(self.input_values[idx])
        item["labels"] = torch.tensor(self.labels[idx])
    
    def __len__(self):
        return len(self.input_values)
    
train_dataset = AudioDataset(train_dataset)
eval_dataset = AudioDataset(eval_dataset)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding : bool
    max_length = None
    pad_to_multiple_of = None

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["label"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(input_features, padding=self.padding, max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        
        batch["labels"] = torch.tensor(label_features, dtype=d_type)


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

if os.path.exists(os.path.join(os.getcwd(), 'wav2vec_model.pickle')):
    with open('wav2vec_model.pickle', 'rb') as file:
        model = pickle.load(file)
else:
    model = Wav2Vec2ForSpeechClassification.from_pretrained("kresnik/wav2vec2-large-xlsr-korean", config=config)
    with open('wav2vec_model.pickle', 'wb') as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)

model.freeze_feature_extractor()

print("Model Loaded")

training_args = TrainingArguments(
    output_dir = "results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=1.0,
    fp16=False,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
)

class CTCTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()