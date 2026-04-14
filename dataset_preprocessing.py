import os 
import pandas as pd
import librosa
import numpy as np

def create_dataset_csv(dataset_path):
    data = []
    # search for all subdirectories (labels) in the dataset path
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        
        # make sure it's a directory (label) and not a file
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith('.wav'):
                    # save the file path and label in the data list
                    file_path = os.path.join(label_path, file_name)
                    data.append([file_path, label])
    
    # create DataFrame
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    
    return df


def audio_to_mel_spectrogram(file_path, n_mels=128, fmax=8000, sr = 44100):
    y, sr = librosa.load(file_path, sr=sr)  # Load audio file
    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)  # Compute Mel spectrogram
    S = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mels, fmax=fmax)  # Compute MFCCs from the Mel spectrogram
    #S_dB = librosa.power_to_db(S, ref=np.max)  # Convert to decibel scale
    
    return S

def convert_audio_to_mel_spectrogram(dataset_path):
    data_mel_spectrogram = []
    for file_path in dataset_path:
        mel_spectrogram = audio_to_mel_spectrogram(file_path)
        data_mel_spectrogram.append(mel_spectrogram)
    
    return data_mel_spectrogram

def normalize_split_data(x_train, y_train, val_size=0.2):
    #normalize the data
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)

    # Split data
    val_size_int = int(np.round(len(x_train))*val_size)
    x_val = x_train[-val_size_int:]
    y_val = y_train[-val_size_int:]
    x_train = x_train[:-val_size_int]
    y_train = y_train[:-val_size_int]

    return x_train, y_train, x_val, y_val