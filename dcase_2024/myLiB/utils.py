# from deepdiff import DeepDiff

# from .models import *

#from .train import *  # necessario o . antes do import
# import skimage.measure
# from featureExtraction import *
# plt.switch_backend('agg')
# from .train import *  # necessario o . antes do import
import logging

import numpy as np
# import skimage.measure
# from featureExtraction import *
# plt.switch_backend('agg')
import pandas as pd
# from .train import *  # necessario o . antes do import
# import skimage.measure
# from featureExtraction import *
# plt.switch_backend('agg')
# from .train import *  # necessario o . antes do import
import logging

import numpy as np
# import skimage.measure
# from featureExtraction import *
# plt.switch_backend('agg')
import pandas as pd

# from deepdiff import DeepDiff
# from .models import *


path_meta = "..\\dcase2022_task1_baseline-main\\dataset\\TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet\\meta.csv"
path_test = "..\\dcase2022_task1_baseline-main\\dataset\\TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet\\evaluation_setup\\fold1_test.csv"
path_train = "..\\dcase2022_task1_baseline-main\\dataset\\TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet\\evaluation_setup\\fold1_train.csv"
path_evaluation = "..\\dcase2022_task1_baseline-main\\dataset\\TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet\\evaluation_setup\\fold1_evaluate.csv"

def int16_to_float32(x):
    import numpy as np
    return (x / 32767.).astype(np.float32)
def float32_to_int16(x):
    if np.max(np.abs(x)) > 1.:
        x /= np.max(np.abs(x))
    return (x * 32767.).astype(np.int16)
def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / len(labels))
    return labels
def Mysmooth_labels(labels):
    scene_labels = ['airport', 'bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

    if ((labels == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all()):  # airport
        labels = [0.98, 0, 0, 0, 0, 0, 0.01, 0.01, 0, 0]

    elif ((labels == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all()):  # bus
        labels = [0, 0.98, 0.01, 0, 0, 0, 0, 0, 0, 0.01]

    elif ((labels == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all()):  # metro
        labels = [0, 0, 0.98, 0.01, 0, 0, 0, 0, 0, 0.01]

    elif ((labels == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all()):  # metro_station
        labels = [0.01, 0, 0.01, 0.97, 0, 0, 0.01, 0, 0, 0]

    elif ((labels == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()):  # park
        labels = [0, 0, 0, 0, 0.99, 0, 0, 0, 0.01, 0]

    elif ((labels == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()):  # public_square
        labels = [0, 0, 0, 0.01, 0, 0.97, 0.01, 0, 0.01, 0]

    elif ((labels == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all()):  # shopping_mall
        labels = [0.01, 0, 0, 0, 0, 0, 0.98, 0.01, 0, 0]

    elif((labels==[0,0,0,0,0,0,0,1,0,0]).all()): #street_pedestrian
        labels=[0.01, 0, 0, 0, 0, 0.01, 0, 0.98, 0, 0]

    elif((labels==[0,0,0,0,0,0,0,0,1,0]).all()): #street_traffic (não alterado)
        labels=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif((labels==[0,0,0,0,0,0,0,0,0,1]).all()): #tram
        labels=[0, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.98]


    return labels
def Mysmooth_labels2(labels):
    scene_labels = ['airport', 'bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

    if ((labels == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all()):  # airport
        labels = [0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0.01, 0.01]

    elif ((labels == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all()):  # bus
        labels = [0.01, 0.93, 0, 0.01,0.01,0.01,0.01, 0.01, 0.01,0] #[0, 0.98, 0.01, 0, 0, 0, 0, 0, 0, 0.01]

    elif ((labels == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all()):  # metro
        labels = [0.01, 0.01, 0.93, 0, 0.01, 0.01, 0.01,0.01,0.01, 0] #[0, 0, 0.98, 0.01, 0, 0, 0, 0, 0, 0.01]

    elif ((labels == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all()):  # metro_station
        labels = [0, 0.01, 0, 0.94, 0.01, 0.01, 0, 0.01, 0.01, 0.01] #[0.01, 0, 0.01, 0.97, 0, 0, 0.01, 0, 0, 0]

    elif ((labels == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()):  # park
        labels = [0.01, 0.01, 0.01, 0.01, 0.92, 0.01, 0.01, 0.01, 0, 0.01] #[0, 0, 0, 0, 0.99, 0, 0, 0, 0.01, 0]

    elif ((labels == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()):  # public_square
        labels = [0.01, 0.01, 0.01, 0, 0.01, 0.94, 0, 0.01, 0, 0.01] # [0, 0, 0, 0.01, 0, 0.97, 0.01, 0, 0.01, 0]

    elif ((labels == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all()):  # shopping_mall
        labels = [0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93, 0, 0.01, 0.01] #[0.01, 0, 0, 0, 0, 0, 0.98, 0.01, 0, 0]

    elif((labels==[0,0,0,0,0,0,0,1,0,0]).all()): #street_pedestrian
        labels=[0, 0.01, 0.01, 0.01, 0.01, 0, 0.01, 0.93, 0.01, 0.01] #[0.01, 0, 0, 0, 0, 0.01, 0, 0.98, 0, 0]

    elif((labels==[0,0,0,0,0,0,0,0,1,0]).all()): #street_traffic (não alterado)
        labels=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    elif((labels==[0,0,0,0,0,0,0,0,0,1]).all()): #tram
        labels=[0.01, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93] #[0, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.98]


    return labels

def create_one_hot_encoding(word, unique_words):
    """Creates an one-hot encoding of the `word` word, based on the\
    list of unique words `unique_words`.
    """
    to_return = np.zeros((len(unique_words)))
    to_return[unique_words.index(word)] = 1
    return to_return
#TODO: juntar estas duas funções index
def get_index(file_list):
    meta_file = pd.read_csv(path_meta, sep='\t')
    name_files = meta_file['filename'].to_list()
    index_file = []
    for file in file_list:
        # TODO: comentei e fiz alterações
        # index_file.append(name_files.index('audio'+file['filename'].split('audio')[1]))
        index_file.append(name_files.index('audio/' + file))  # meu update
    return index_file
def get_index2(file_list):
    meta_file = pd.read_csv(path_meta, sep='\t')
    name_files = meta_file['filename'].to_list()
    index_file = []
    for file in file_list:
        # TODO: comentei e fiz alterações
        # index_file.append(name_files.index('audio'+file['filename'].split('audio')[1]))
        index_file.append(name_files.index('audio' + str(file['filename'].split('audio')[1]).replace("\\" or "//", "/")))  # meu update
    return index_file

def get_data(hdf5_path, index_files):
    # Loop through all test files from the current cross-validation fold
    import h5py
    with h5py.File(hdf5_path, 'r') as hf:
        # features = int16_to_float32(hf['features'][index_files])
        features = hf['features'][index_files]
        labels = [f.decode() for f in hf['scene_label'][index_files]]
        audio_name = [f.decode() for f in hf['filename'][index_files]]

    return features, labels, audio_name
def labelsEncoding(split,unique_words,labels):
    Y = []
    for lab in labels:
        if split == 'Train':
            Y.append(smooth_labels(create_one_hot_encoding(lab, unique_words)))
        elif split == 'Mysmooth_labels':
            Y.append(Mysmooth_labels(create_one_hot_encoding(lab, unique_words)))
        elif split == 'Mysmooth_labels2':
            Y.append(Mysmooth_labels2(create_one_hot_encoding(lab, unique_words)))
        else:
            Y.append(create_one_hot_encoding(lab, unique_words))
    return Y
def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)

    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

