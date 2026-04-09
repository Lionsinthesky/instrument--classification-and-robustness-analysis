import os 
import pandas as pd
import librosa

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
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)  # Compute Mel spectrogram
    S_dB = librosa.power_to_db(S, ref=np.max)  # Convert to decibel scale
    
    return S_dB

