
import os
import sys
dir_path = '/home/lab/DCASE/dcase_2024'
sys.path.append(dir_path)   

import time
from concurrent.futures import ThreadPoolExecutor
import dcase_util
import h5py
import librosa
import matplotlib.pyplot as plt

from myLiB.utils import *

scene_labels = {
    'airport':0,
    'bus':1,
    'metro':2,
    'metro_station':3,
    'park':4,
    'public_square':5,
    'shopping_mall':6,
    'street_pedestrian':7,
    'street_traffic':8,
    'tram':9
}


#TODO: ficheiro responsavel pelo pipline de extração de features/data_augmentation e armazenamento de variaveis no h5py prontas a treinar
print("Extração de features")


def preprocess_audio(file_path):
    # Load audio file with librosa
    y, sr = librosa.load(file_path, sr=48000)
    # Convert to mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize
    mel_spec_db_norm = (mel_spec_db + 80) / 80

    return mel_spec_db_norm



# Setup logging
#path_meta = dir_path + "/Dataset/meta.csv"
#path_train = dir_path + "/Dataset/evaluation_setup/fold1_train.csv"
path_train = dir_path + "/fold1_val.csv"

datasetPathAudio = dir_path + "/Dataset/TAU-urban-acoustic-scenes-2022-mobile-development"


train_file = pd.read_csv(path_train, sep=',')
train_filename = np.array(train_file['filename'])
#train_scene_label=train_file['scene_label']
train_scene_label = []
for label in train_file['scene_label']:
    train_scene_label.append(scene_labels[label])

i = 0

# Define the save path
save_path = dir_path  + "/Archive_val"


# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

for audio_filename in enumerate(train_filename):
    path = datasetPathAudio + "/" + audio_filename[1]
    mel_spectrogram = preprocess_audio(path)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=48000, x_axis='time', y_axis='mel')
    plt.axis('off')  # No axis for image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    file_name = audio_filename[1].split('/')
    ggg = save_path  + "/" + file_name[1] + '.png'

    plt.savefig(ggg, bbox_inches='tight', pad_inches=0)
    print(str(i) + ". Saved at " + ggg)
    plt.close()
    i = i + 1


    

