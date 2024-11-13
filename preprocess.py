# Define functions to preprocess audio files

# 1. few_second_clip
## Input: folder with audio files
## User defined: clip duration and working directory.
## Output: new folder with clipped audio filies labelled with suffix '_clip_n' where n is 0,1,2...

# 2. audio_process
## Input: folder with audio files
## User defined: bandpass filter frequencies and sampling rate
## Output: new folder with processed audio files labelled with suffic '_f'

# 3. spec_process
## Input: folder with audio files
## User defined: fft_n and hop size, option to change midpoint of color map
## Output: new folder with spectograms labelled with prefix 's_'

####################################################

import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf
import librosa.display
import matplotlib.pylab as plt
from scipy.signal import butter, lfilter, freqz
from scipy.io.wavfile import write
import imageio.v3 as iio
import matplotlib.colors as colors

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

# 2. Load audio as mono file, resample and apply band pass filter function

def audio_process(td, 
                 sf = 22050,
                 cutoff=[50,1500], order=5):
    # td: location of all audio files
    # sr: required sampling rate in Hz, defaults to 22.05kHz
    # cutoff: band pass frequency in Hz, defaults to [0, 1500]
    # order: order of butterworth filter, defaults to 5
    
    # Create folders to store processed audio
    filepath = f'{td}filtered\\'
    if not os.path.exists(filepath): os.makedirs(filepath)
    
    # For all .wav files in the defined working directory
    files = [f for f in os.listdir(td) if os.path.isfile(f'{td}{f}') and f.endswith(".wav")]
    
    for name in files:
        # Load audio in mono and resample
        aud, sr = librosa.load(f'{td}{name}', sr=sf, mono=True)

        # Define prefix to name output files
        filename = name[:-4] 
        
        # Generate butterworth coefficients a,b
        a,b = butter(N= order, Wn = cutoff, fs=sr, btype='band', analog=False)
        # Apply filter that returns filtered audio y
        y = lfilter(a,b, aud)
        # Save to wav file
        write(f'{filepath}{filename}_f.wav', rate = sr, data = y.astype(np.float32))
    
    # Display success message
    print(f'Filtered audio for {td}')
    
####################################################

# Create spectrogram with given fft_n and hop_size
# # Optional- change midpoint of colormap

def spec_process(td,
                 fft_n = 256, hop_size = 128, mid=50):
    # td: location of all audio files
    # fft_n: fast fourier transform window length
    # hop_size: hop length
    
    min_dB = 0
    max_dB = 100
    
    # Create folders to store spectrograms
    filepath = f'{td}specs\\'
    if not os.path.exists(filepath): os.makedirs(filepath)
    
    # For all .wav files in the defined working directory
    files = [f for f in os.listdir(td) if os.path.isfile(f'{td}{f}') and f.endswith(".wav")]
    
    for name in files:
        # Load audio
        aud, sr = librosa.load(f'{td}{name}', sr=None)

        # calculate short time FT
        aud_sfft = librosa.stft(aud, n_fft=fft_n, hop_length=hop_size)

        # Converts stft output of complex numbers into abs no for better visualization
        aud_y = np.abs(aud_sfft)**2 

        # convert from power to dB 
        aud_y_log = librosa.power_to_db(aud_y)

        # Normalize the spectrogram to the range min and max dB
        # Get the min and max of the original dB spectrogram
        aud_y_log_min = np.min(aud_y_log)
        aud_y_log_max = np.max(aud_y_log)

        # Rescale the dB spectrogram to the desired range
        aud_y_log_norm = min_dB + ((aud_y_log - aud_y_log_min) / (aud_y_log_max - aud_y_log_min)) * (max_dB - min_dB)
        
        plt.figure(figsize=(15.55,3.11))  # 3.11, 3.11 creates a plot that will result in a 224x224 resolution (224 px / 72 DPI = ~3.11 inches)
        
        if (mid==50): # if no value specified, make default spectrogram
            librosa.display.specshow(aud_y_log_norm, sr=sr,hop_length=hop_size, y_axis='log', cmap='viridis')
        else: # if value specified, then use
            ## Define the normalization mid point for cmap
            diversity_norm = colors.TwoSlopeNorm(vcenter=mid)
            librosa.display.specshow(aud_y_log_norm, sr=sr,hop_length=hop_size, y_axis='log', cmap='viridis',norm = diversity_norm)
        
        plt.rcParams.update({'font.size': 0})
        plt.yticks([]) # Remove y ticks
        plt.xticks([]) # Remove x ticks  
        plt.savefig(f'{filepath}s_{name[:-4]}.png')
        plt.clf()
        plt.close()
    
    # Display success message
    print(f'Created specs for audio in {td}')

####################################################



