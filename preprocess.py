import numpy as np
import torch
from tqdm import tqdm
import pickle

import os
import csv

emotions = ["surprise", "fear", "angry", "neutral", "sad", "happy", "disgust"]

def emot_num(emo):
    return emotions.index(emo)


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

for file_name in tqdm(annotation_csv_files):
    with open(os.path.join(path_name, annotation_dir, file_name), newline='') as file:
        session_num = int(file_name.split("_")[0][-2:])
        session_gen = file_name.split("_")[1]

        reader = csv.reader(file)
        iterate = 0
        for row in reader:
            iterate += 1
            if(iterate < 3):
                continue

            sample_point = {}
            sample_point["annotation_name"] = file_name
            sample_point["session_num"] = session_num
            sample_point["session_gen"] = session_gen

            sample_point["annotation_number"] = row[0]
            sample_point["wav_start"] = float(row[1])
            sample_point["wav_end"] = float(row[2])

            sample_point["ecg_start"] = float(row[3]) if row[3] != '' else None
            sample_point["ecg_end"] = float(row[4]) if row[4] != '' else None
            sample_point["eda_start"] = float(row[5]) if row[5] != '' else None
            sample_point["eda_end"] = float(row[6]) if row[6] != '' else None
            sample_point["temp_start"] = float(row[7]) if row[7] != '' else None
            sample_point["temp_end"] = float(row[8]) if row[8] != '' else None

            sample_point["segment_id"] = row[9]
            sample_point["total_emot"] = [emot_num(x) for x in row[10].split(";")]
            sample_point["total_valence"] = float(row[11])
            sample_point["total_arousal"] = float(row[12])

            sample_point["eval_emot"] = [0,0,0,0,0,0,0]
            sample_point["eval_valence"] = [0,0,0,0,0]
            sample_point["eval_arousal"] = [0,0,0,0,0]

            for i in range(13, 42, 3):
                sample_point["eval_emot"][emot_num(row[i])] += 1
                sample_point["eval_valence"][int(row[i+1])-1] += 1
                sample_point["eval_arousal"][int(row[i+2])-1] += 1
            
            wav_file_dir = "dataset/KEMDy19/wav/Session" + str(session_num).zfill(2) + "/" + sample_point["segment_id"][:-5]
            
            try:
                with open(os.path.join(path_name, wav_file_dir, sample_point["segment_id"] + ".txt"), "r") as txt_file:
                    sample_point["text"] = txt_file.read()[:-1]
            except FileNotFoundError:
                sample_point["text"] = None

            # sample_point["wav"] = ...
            # sample_point["edg"] = ...
            # sample_point["temp"] = ...

            sample_name = "dataset/processed/" +str(session_num).zfill(2) + "_" + session_gen + "_" + sample_point["segment_id"]

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)