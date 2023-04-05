#!/bin/bash

# Specify the folder path
folder="KEMDy19/wav/Session01/Sess01_impro01"

# Loop through all files in the folder
for file in "$folder"/*; do
    # Check if the current item is a file
    if [ ${file: -4} == ".wav" ]; then
        echo "${file}";
        python classify.py --audio ${file} --class_name class_name.txt --model large --language Korean --device cpu
        # Add your desired commands or actions here for each file
    fi
done

