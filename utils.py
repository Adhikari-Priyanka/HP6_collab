# Utility functions

# 1. filename_to_csv
## Input: folder with audio or spectrograms
## User defined: file format and whether to record sampling rate also
## Output: csv file with names of all files in folder

# 2. move_to_folder
## Input: folder with audio or spectrograms
## User defined: target folder to move files to and format of the files
## Output: files moved to target

# 3. move_to_label
## Input: folder with audio or spectrograms and a csv containing 'filename' and 'label' columns
## User defined: target folder and format of the files
## Output: Within target folder, files are moved to folders corresponding to labels in the csv

# 4. combin
## Input: folder with audio files
## Output: Combined audio file in same folder

####################################################

import numpy as np
import os
import pandas as pd
import librosa
from pydub import AudioSegment

####################################################

# 1. Define function to read filenames and save to a csv.
## For .wav files, also save sampling rates

def filename_to_csv(td, format='.wav'):
    # td: location of all audio files
    # format : format of the files '.wav' or .png'
    # sr= True or False- if sampling rate value is also needed
    
    # Generate a pandas series of all filenames ending with .wav
    files = pd.Series([f for f in os.listdir(td) if os.path.isfile(f'{td}{f}') and f.endswith(format)])

    # Export the df as csv
    files.to_csv(f'{td}file_names.csv', index=True)
    
    # Display success message
    print(f'Filename csv created for {td}')

####################################################

# 2. Define function to move files from source folder to target folder

def move_to_folder(source, target, format = '.wav'):
    # source: source folder containing all format files
    # target: target folder to move files into
    
    # Create target folder if not already present
    if not os.path.exists(target): os.makedirs(target) 
    
    # read names of all format files in the source folder
    files = [f for f in os.listdir(source) if os.path.isfile(f'{source}{f}') and f.endswith(format)]
    
    for n in files:
        # move to new location
        os.replace(f'{source}{n}' ,f'{target}{n}') 
    
    # Display success message
    print(f'{format} files moved from {source} to {target}')

####################################################

# 3. Function to move files based on label

def move_to_label(source, target, df, format='.wav'):
    # source: source folder containing all audio files
    # target: target folder to create folders corresponding to labels and move files to
    # df: path to csv containing 'filename' including '.wav' and 'label'
    # format :type  of file eg '.wav' or '.png
    
    # Read csv file
    lab_df = pd.read_csv(df)
    
    # Create folder in target folder corresponding to labels in csv
    labels = pd.Series([lab_df['label'].unique()][0])
    for i in labels:
        filepath = f'{target}\\{i}\\'
        if not os.path.exists(filepath): os.makedirs(filepath) 
        # Display success message
        print(f'Folder created for label {i}')
    
    # read names of all files in source
    files = [f for f in os.listdir(source) if os.path.isfile(f'{source}{f}') and f.endswith(format)]
    
    for n in files:
        # Select the label that corresponds to filename 'n'
        n_lab = lab_df[lab_df['filename'] == n]['label'].tolist()[0] # [1] selects the specific string

        # New destination path
        dest = f'{target}{n_lab}\\'
        
        # Move to new
        os.replace(f'{source}{n}', f'{dest}{n}')
        
        # Display success message
        print(f'{format} files moved from {source} to {target}')

####################################################

# 4. Function to combine all audio files in a folder into a single audio file

def combin(td):
    # td: name of the folder with all your wav files
    
    # read names of all '.wav' files in td
    files = [f for f in os.listdir(td) if os.path.isfile(f'{td}{f}') and f.endswith('.wav')]
    
    # Get the first wav file
    sound_append = AudioSegment.from_wav(f'{td}{files[0]}')
    
    # For loop to take all other files and append
    for i in range(1, len(files)):
        # Take the next sound 
        sound = AudioSegment.from_wav(f'{td}{files[i]}')
        # Update sound_append 
        sound_append = sound_append + sound
    
    # Export combine wav file
    sound_append.export(f'{td}combined.wav', format= 'wav')
    
    # Print success message
    print(f'{td} done')
    
    ####################################################

####################################################

