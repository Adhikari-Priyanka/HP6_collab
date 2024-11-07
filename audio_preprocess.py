# Define functions to preprocess audio files

# 1. few_second_clip .wav files
## Input: folder with audio files
## User defined: clip duration and working directory.
## Output: new folder with clipped audio filies labelled with suffix _clip_n where n is 0,1,2...

####################################################

import pandas as pd
import numpy as np
import os
from pydub import AudioSegment
import librosa
import soundfile as sf
import librosa.display
import IPython.display as ipd
import matplotlib.pylab as plt
from scipy.signal import butter, lfilter, freqz
from scipy.io.wavfile import write

####################################################

# 1. few_second_clip NOAA files

def few_second_clips(td, clip = 5, overlap = 1):
    # td : location of the audio file
    # clip : maximum duration of each clip
    
    # Define file pah to store outputs and create if not already present
    filepath = f'{td}\\{clip}s_clip\\'
    if not os.path.exists(filepath): os.makedirs(filepath) 
    
    # Read names of all .wav files in the defined working directory, clip as per csv file and export clipped file
    files = [f for f in os.listdir(td) if os.path.isfile(f'{td}{f}') and f.endswith('.wav')]
    
    # for loop to clip all the audio files
    for n in files:
        name = n[:-4] # Extract just name of the file not '.wav'
    
        # Load audio file
        aud, sr = librosa.load(f'{td}{name}.wav')
        # Find duration of the audio
        dur = int(len(aud)/sr)
                
        # Precompute all slice points at once
        clip_times = [(i, i + clip) for i in range(0, dur, int(clip*overlap))]
        # Process each slice
        for start_time, end_time in clip_times:
            # Ensure end_time does not exceed the duration
            if end_time > dur:
                end_time = dur

            # Slice audio
            clip_audio = aud[start_time*sr:end_time*sr]
            
            # Export clipped audio
            sf.write(f'{filepath}{name}_{start_time}.wav', clip_audio, sr)

        # Display success message
        print(f'Audio: {name} clipped')

####################################################

# 2. Low pass filter function

def filter_audio(td, cutoff=1500, order=5):
    # td: location of all audio files
    # cutoff: low pass frequency in Hz
    # order: order of butterworth filter
    
    # Create folders to store clipped audio and noise
    filepath = f'{td}filtered\\'
    if not os.path.exists(filepath): os.makedirs(filepath)
    
    # For all .wav files in the defined working directory
    files = [f for f in os.listdir(td) if os.path.isfile(f'{td}{f}') and f.endswith(".wav")]
    
    for name in files:
        # Load audio
        aud, sr = librosa.load(f'{td}{name}')

        # Define prefix to name output files
        filename = name[:-4] 
        
        # Generate butterworth coefficients a,b
        a,b = butter(order, cutoff, fs=sr, btype='low', analog=False)
        # Apply filter that returns filtered audio y
        y = lfilter(a,b, aud)
        # Save to wav file
        write(f'{filepath}{filename}_f.wav', rate = sr, data = y.astype(np.float32))
    
    # Display success message
    print(f'Filtered audio for {td}')