from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, Wav2Vec2Processor
import torchaudio

import os
import csv

emotions = ["surprise", "fear", "angry", "neutral", "sad", "happy", "disgust"]


def emot_num(emo):
    if emo == "disqust":
        return 6
    if emo == "anger":
        return 2
    if emo == "sadness":
        return 4
    if emo == "happiness":
        return 5
    return emotions.index(emo)


# Delete Current Processed Files
processed_dir = "dataset/processed"

for f in tqdm(os.listdir(processed_dir)):
    os.remove(os.path.join(processed_dir, f))

############# Annotation Details #################
# 0 : Number
# 1,2 : Wav Start, End
# 3,4 : ECG Start, End
# 5,6 : E4-EDA Start, End
# 7,8 : E4-Temp Start, End
# 9 : Segment ID
# 10,11,12 : Total Eval Emotion,Valence,Arousal
# 13~42 : All Eval Emotion,Valence,Arousal
##################################################

path_name = os.getcwd()
annotation_dir = "dataset/KEMDy19/annotation"

annotation_csv_files = []

file_names_annotation = os.listdir(os.path.join(path_name, annotation_dir))
for file_name in file_names_annotation:
    if file_name.endswith(".csv"):
        annotation_csv_files.append(file_name)

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
tokenizer.add_tokens(["c/", "n/", "N/", "u/", "l/", "b/", "*", "+", "/"])

if os.path.exists(os.path.join(os.getcwd(), 'wav2vec_processor.pickle')):
    with open('wav2vec_processor.pickle', 'rb') as file:
        processor = pickle.load(file)
else:
    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    with open('wav2vec_processor.pickle', 'wb') as file:
        pickle.dump(processor, file, pickle.HIGHEST_PROTOCOL)

target_sampling_rate = processor.feature_extractor.sampling_rate

for file_name in tqdm(annotation_csv_files):
    with open(os.path.join(path_name, annotation_dir, file_name), newline='') as file:
        session_num = int(file_name.split("_")[0][-2:])
        session_gen = file_name.split("_")[1]

        reader = csv.reader(file)
        iterate = 0
        for row in reader:
            iterate += 1
            if iterate < 3:
                continue
            if row[9].split("_")[2][0] != session_gen:
                continue

            sample_point = {}

            sample_point["segment_id"] = row[9]
            sample_point["total_emot"] = [emot_num(x) for x in row[10].split(";")]
            sample_point["total_valence"] = float(row[11])
            sample_point["total_arousal"] = float(row[12])

            wav_file_dir = "dataset/KEMDy19/wav/Session" + str(session_num).zfill(2) + "/" + sample_point["segment_id"][:-5] + "/" + sample_point["segment_id"] + ".wav"
            sample_point["wav_dir"] = wav_file_dir

            speech_array, sampling_rate = torchaudio.load(wav_file_dir)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()

            speech_feature = processor.__call__(audio=speech, sampling_rate=target_sampling_rate)

            sample_point["wav_input_values"] = speech_feature['input_values'][0]
            sample_point["wav_attention_mask"] = speech_feature['attention_mask'][0]

            wav_file_dir = "dataset/KEMDy19/wav/Session" + str(session_num).zfill(2) + "/" + sample_point["segment_id"][:-5]

            try:
                with open(os.path.join(path_name, wav_file_dir, sample_point["segment_id"] + ".txt"), "r", encoding="UTF-8") as txt_file:
                    text = txt_file.read()[:-1]
                    inputs = tokenizer(
                                text,
                                return_tensors='pt',
                                truncation=True,
                                max_length=256,
                                pad_to_max_length=True,
                                add_special_tokens=True
                                )
                    input_ids = inputs['input_ids'][0]
                    attention_mask = inputs['attention_mask'][0]

                    sample_point["input_ids"] = input_ids
                    sample_point["attention_mask"] = attention_mask
                    sample_point["text"] = text
            except FileNotFoundError:
                sample_point["text"] = None

            sample_name = "dataset/processed/K19_" + str(session_num).zfill(2) + "_" + session_gen + "_" + sample_point["segment_id"]

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)


############# Annotation Details #################
# 0 : Number
# 1,2 : Wav Start, End
# 3 : Segment ID
# 4,5,6 : Total Eval Emotion,Valence,Arousal
# 7~36 : All Eval Emotion,Valence,Arousal
##################################################

path_name = os.getcwd()
annotation_dir = "dataset/KEMDy20/annotation"

annotation_csv_files = []

file_names_annotation = os.listdir(os.path.join(path_name, annotation_dir))
for file_name in file_names_annotation:
    if file_name.endswith(".csv"):
        annotation_csv_files.append(file_name)

for file_name in tqdm(annotation_csv_files):
    with open(os.path.join(path_name, annotation_dir, file_name), newline='') as file:
        session_num = int(file_name.split("_")[0][-2:])

        reader = csv.reader(file)
        iterate = 0
        for row in reader:
            iterate += 1
            if(iterate < 3):
                continue

            sample_point = {}

            sample_point["segment_id"] = row[3]
            sample_point["total_emot"] = [emot_num(x) for x in row[4].split(";")]
            sample_point["total_valence"] = float(row[5])
            sample_point["total_arousal"] = float(row[6])

            wav_file_dir = "dataset/KEMDy20/wav/Session" + str(session_num).zfill(2) + "/" + sample_point["segment_id"] + ".wav"
            sample_point["wav_dir"] = wav_file_dir

            speech_array, sampling_rate = torchaudio.load(wav_file_dir)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()

            speech_feature = processor.__call__(audio=speech, sampling_rate=target_sampling_rate)

            sample_point["wav_input_values"] = speech_feature['input_values'][0]
            sample_point["wav_attention_mask"] = speech_feature['attention_mask'][0]
            
            wav_file_dir = "dataset/KEMDy20/wav/Session" + str(session_num).zfill(2)
            
            try:
                with open(os.path.join(path_name, wav_file_dir, sample_point["segment_id"] + ".txt"), "r") as txt_file:
                    text = txt_file.read()[:-1]
                    inputs = tokenizer(
                                text,
                                return_tensors='pt',
                                truncation=True,
                                max_length=256,
                                pad_to_max_length=True,
                                add_special_tokens=True
                                )
                    input_ids = inputs['input_ids'][0]
                    attention_mask = inputs['attention_mask'][0]

                    sample_point["input_ids"] = input_ids
                    sample_point["attention_mask"] = attention_mask
                    sample_point["text"] = text
            except FileNotFoundError:
                sample_point["text"] = None

            sample_name = "dataset/processed/K20_" +str(session_num).zfill(2) + "_" + session_gen + "_" + sample_point["segment_id"]

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)

############# Annotation Details #################
# 0 : wav_id
# 1 : text
# 2 : emotions
# 3~12 : eval emotion
# 13 : age
# 14: gender
##################################################

path_name = os.getcwd()
annotation_dir = "dataset/other_set/year_4/annotation"

annotation_csv_files = []

file_names_annotation = os.listdir(os.path.join(path_name, annotation_dir))
for file_name in file_names_annotation:
    if file_name.endswith(".csv"):
        annotation_csv_files.append(file_name)


for file_name in tqdm(annotation_csv_files):
    with open(os.path.join(path_name, annotation_dir, file_name), newline='') as file:

        reader = csv.reader(file)
        iterate = 0

        for row in reader:
            iterate += 1
            if iterate < 2:
                continue
            sample_point = {}

            sample_point["segment_id"] = row[0]
            sample_point["total_emot"] = [emot_num(x) for x in row[2].split(";")]

            wav_file_dir = "dataset/other_set/year_4/year_4_wav" + "/" + sample_point["segment_id"] + ".wav"
            sample_point["wav_dir"] = wav_file_dir

            try:
                with open(wav_file_dir, "r") as wav_file:
                    nothing = 1
            except FileNotFoundError:
                continue

            speech_array, sampling_rate = torchaudio.load(wav_file_dir)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()

            speech_feature = processor.__call__(audio=speech, sampling_rate=target_sampling_rate)

            sample_point["wav_input_values"] = speech_feature['input_values'][0]
            sample_point["wav_attention_mask"] = speech_feature['attention_mask'][0]

            inputs = tokenizer(
                                row[1],
                                return_tensors='pt',
                                truncation=True,
                                max_length=256,
                                pad_to_max_length=True,
                                add_special_tokens=True
                                )

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            sample_point["input_ids"] = input_ids
            sample_point["attention_mask"] = attention_mask
            sample_point["text"] = row[1]

            sample_name = "dataset/processed/yr4_50" + "_" + sample_point["segment_id"]

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)


############# Annotation Details #################
# 0 : wav_id
# 1 : text
# 2 : emotions
# 3~12 : eval emotion
# 13 : age
# 14: gender
##################################################

path_name = os.getcwd()
annotation_dir = "dataset/other_set/year_5_1/annotation"

annotation_csv_files = []

file_names_annotation = os.listdir(os.path.join(path_name, annotation_dir))
for file_name in file_names_annotation:
    if file_name.endswith(".csv"):
        annotation_csv_files.append(file_name)


for file_name in tqdm(annotation_csv_files):
    with open(os.path.join(path_name, annotation_dir, file_name), newline='') as file:

        reader = csv.reader(file)
        iterate = 0
        for row in reader:
            iterate += 1
            if iterate < 2:
                continue

            sample_point = {}

            sample_point["segment_id"] = row[0]
            sample_point["total_emot"] = [emot_num(x) for x in row[2].split(";")]

            wav_file_dir = "dataset/other_set/year_5_1/year_5_1_wav" + "/" + sample_point["segment_id"] + ".wav"
            sample_point["wav_dir"] = wav_file_dir

            try:
                with open(wav_file_dir, "r") as wav_file:
                    nothing = 1
            except FileNotFoundError:
                continue

            speech_array, sampling_rate = torchaudio.load(wav_file_dir)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()

            speech_feature = processor.__call__(audio=speech, sampling_rate=target_sampling_rate)

            sample_point["wav_input_values"] = speech_feature['input_values'][0]
            sample_point["wav_attention_mask"] = speech_feature['attention_mask'][0]
            
            inputs = tokenizer(
                                row[1],
                                return_tensors='pt',
                                truncation=True,
                                max_length=256,
                                pad_to_max_length=True,
                                add_special_tokens=True
                                )
            
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            sample_point["input_ids"] = input_ids
            sample_point["attention_mask"] = attention_mask
            sample_point["text"] = row[1]

            sample_name = "dataset/processed/y51_50" + "_" + sample_point["segment_id"]

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)


############# Annotation Details #################
# 0 : wav_id
# 1 : text
# 2 : emotions
# 3~12 : eval emotion
# 13 : age
# 14: gender
##################################################

path_name = os.getcwd()
annotation_dir = "dataset/other_set/year_5_2/annotation"

annotation_csv_files = []

file_names_annotation = os.listdir(os.path.join(path_name, annotation_dir))
for file_name in file_names_annotation:
    if file_name.endswith(".csv"):
        annotation_csv_files.append(file_name)


for file_name in tqdm(annotation_csv_files):
    with open(os.path.join(path_name, annotation_dir, file_name), newline='') as file:

        reader = csv.reader(file)
        iterate = 0
        for row in reader:
            iterate += 1
            if iterate < 2:
                continue

            sample_point = {}

            sample_point["segment_id"] = row[0]
            sample_point["total_emot"] = [emot_num(x) for x in row[2].split(";")]

            wav_file_dir = "dataset/other_set/year_5_2/year_5_2_wav" + "/" + sample_point["segment_id"] + ".wav"
            sample_point["wav_dir"] = wav_file_dir

            try:
                with open(wav_file_dir, "r") as wav_file:
                    nothing = 1
            except FileNotFoundError:
                continue

            speech_array, sampling_rate = torchaudio.load(wav_file_dir)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()

            speech_feature = processor.__call__(audio=speech, sampling_rate=target_sampling_rate)

            sample_point["wav_input_values"] = speech_feature['input_values'][0]
            sample_point["wav_attention_mask"] = speech_feature['attention_mask'][0]

            inputs = tokenizer(
                                row[1],
                                return_tensors='pt',
                                truncation=True,
                                max_length=256,
                                pad_to_max_length=True,
                                add_special_tokens=True
                                )

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            sample_point["input_ids"] = input_ids
            sample_point["attention_mask"] = attention_mask
            sample_point["text"] = row[1]

            sample_name = "dataset/processed/y52_50" + "_" + sample_point["segment_id"]

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)
