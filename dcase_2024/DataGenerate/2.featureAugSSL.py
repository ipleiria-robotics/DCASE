import os
import sys
dir_path = '/home/lab/DCASE/dcase_2024'
sys.path.append(dir_path)   

import time
from concurrent.futures import ThreadPoolExecutor
import dcase_util
import h5py
import librosa

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

print("Extração de features")


# Setup logging
path_meta = dir_path + "/Dataset/meta.csv"
path_test = dir_path + "/Dataset/evaluation_setup/fold1_test.csv"
path_train = dir_path + "/Dataset/evaluation_setup/fold1_train.csv"
path_evaluation = dir_path + "/Dataset/evaluation_setup/fold1_evaluate.csv"

path_SSL = dir_path + "/Dataset/evaluation_setup/fold1_query_index.csv"


#path_SSL = dir_path + "/Dataset/evaluation_setup/fold1_train.csv"

datasetPathAudio = dir_path + "/Dataset/TAU-urban-acoustic-scenes-2022-mobile-development"


output_dir=""

#Baseline Settings (44.1kHz) (51mels) (0.04s win) (0.02s hop) | SHAPE(51,51)
fs_new=44100 # 7000 #28000
fs_Orig=44100 
dicc = {'spectrogram_type': 'magnitude',
        'hop_length_seconds': 0.02,   #0.02 
        'win_length_seconds': 0.04,  #0.04
        'window_type': 'hamming_asymmetric',
        'n_mels': 40,
        'n_fft': 2048,
        'fmin': 0,
        'fmax': fs_Orig // 2,  #22050
        'htk': False,
        'normalize_mel_bands': False,
        'method': 'mel',
        'fs': fs_Orig}

dataAugmentationAudio=False
dataAugmentationSpec=False
normalizar=False

extractor = dcase_util.features.MelExtractor(**dicc)


testQuery='QueryIndex_fs' + str(fs_new) + "_" + str(dicc['n_mels']) + "_" + str(dicc['n_fft']) + "_" + str(dicc['win_length_seconds']) + "_" + str(dicc['hop_length_seconds']) + ".h5"

descricao="\nfs" + str(fs_new) + " n_fft_" + str(dicc['n_fft']) + " n_mels_" + str(dicc['n_mels']) \
          +"\nwin_length_" + str(dicc['win_length_seconds']) +" hop_length_" + str(dicc['hop_length_seconds']) +\
          "\nDaugAudio_" + str(dataAugmentationAudio) +"\nDaugSpec_" + str(dataAugmentationSpec)+"\nNormalize_" + str(normalizar)



################################################# META.CSV #############################################################
audios_dir = pd.read_csv(path_meta, sep='\t', index_col=False)
audio_names = audios_dir['filename']
audio_names_List = audios_dir['filename'].tolist() #audio_names em formato list
#audio_labels = audios_dir['scene_label']
audio_labels = []
for label in audios_dir['scene_label']:
    audio_labels.append(scene_labels[label])

newData_identifier = audios_dir['identifier']
source_label = audios_dir['source_label']



########################################################################################################################
################################################# Query.CSV #############################################################

train_file = pd.read_csv(path_SSL, sep='\t')
train_filename = np.array(train_file['filename'])
#train_scene_label=train_file['scene_label']
train_scene_label = []
for label in train_file['scene_label']:
    train_scene_label.append(scene_labels[label])

train_source_label=[]
train_indentifier=[]
# train_scene_label2=[]
start_time = time.time()
for item_id, audio_filename in enumerate(train_filename):
    try:
        idx=audio_names_List.index(audio_filename)  #procurar audio de train no meta.csv para saber source,identifier ...
        train_source_label.append(source_label[idx])
        train_indentifier.append(newData_identifier[idx])
        # train_scene_label2.append(audio_labels[idx])
    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))


data = np.array([train_filename,train_scene_label, train_source_label,train_indentifier]).T
df = pd.DataFrame(data, columns=['filename','scene_label', 'source_label','identifier'])
df['Na']=(df['source_label']=='a')  #Encontrar audios do tipo A depois ignorados no dataAugmetation  PAG66 -> Explicação -> existem mais dados do dispositivo A do que no restante 
del train_filename,train_scene_label,train_source_label,train_indentifier,data,source_label  #libertar espaço RAM

start_time = time.time()
newData=[]
newData_scene_label=[]
newData_source_label=[]
newData_identifier=[]
yy = 0
with ThreadPoolExecutor() as executor:
    for i in range(0,len(df['filename'])):
        audio = dcase_util.containers.AudioContainer().load(filename=datasetPathAudio + "/" + df['filename'][i], mono=True, fs=fs_Orig)
        audio=audio.data
        if (fs_Orig != fs_new):
            audio = librosa.resample(audio, orig_sr=fs_Orig, target_sr=fs_new)
        audio = np.float32(audio)
        spec = dcase_util.containers.FeatureContainer(data=extractor.extract(audio))
        spec=spec.data
        newData.append(spec)
        newData_scene_label.append(df['scene_label'][i])
        newData_source_label.append(df['source_label'][i])
        newData_identifier.append(df['identifier'][i])
        yy = yy + 1
        print(yy)
     


for i in range(len(newData_identifier)):
        if ('london' in newData_identifier[i]):
            newData_identifier[i] = 'london'
        elif ('barcelona' in newData_identifier[i]):
            newData_identifier[i] = 'barcelona'
        elif ('helsinki' in newData_identifier[i]):
            newData_identifier[i] = 'helsinki'
        elif ('stockholm' in newData_identifier[i]):
            newData_identifier[i] = 'stockholm'
        elif ('milan' in newData_identifier[i]):
            newData_identifier[i] = 'milan'
        elif ('vienna' in newData_identifier[i]):
            newData_identifier[i] = 'vienna'
        elif ('lyon' in newData_identifier[i]):
            newData_identifier[i] = 'lyon'
        elif ('paris' in newData_identifier[i]):
            newData_identifier[i] = 'paris'
        elif ('lisbon' in newData_identifier[i]):
            newData_identifier[i] = 'lisbon'
        elif ('prague' in newData_identifier[i]):
            newData_identifier[i] = 'prague'


from sklearn.model_selection import train_test_split

#query data 50% Train 50%index
X_query, X_index, Y_query, Y_index = train_test_split(newData, newData_scene_label, test_size=0.5,random_state=1, stratify=newData_identifier)
del newData,newData_scene_label,newData_identifier,newData_source_label

packed_hdf5_path = os.path.join(output_dir,testQuery)
meta_dict = {
    'X_query': np.array(X_query),
    'X_index': np.array(X_index),
    'Y_query': np.array(Y_query),
    'Y_index': np.array(Y_index),
    'descricao': np.array(descricao),
}

audios_numtrain = len(meta_dict['Y_query'])
audios_numval = len(meta_dict['Y_index'])

with h5py.File(packed_hdf5_path, 'w') as hf:
    hf.create_dataset(
        name='Y_query',
        shape=(audios_numtrain,),
        dtype='int64') # 'S20'

    hf.create_dataset(
        name='Y_index',
        shape=(audios_numval,),
        dtype='int64')  # 'S20'

    hf.create_dataset(
        name='X_query',
        shape=(len(X_query), X_query[0].shape[0],X_query[0].shape[1]),
        dtype=np.float32) # 'uint8'

    hf.create_dataset(
        name='X_index',
        shape=(len(X_index), X_index[0].shape[0],X_index[0].shape[1]),
        dtype=np.float32)

    hf.create_dataset(
        name='descricao',
        shape=(len(descricao.encode('utf-8')),),
        dtype='S120')

    for n in range(len(descricao.encode('utf-8'))):
        hf['descricao'][n] = descricao[n].encode('utf-8')

    for n in range(audios_numtrain):
        hf['X_query'][n] = X_query[n]
        hf['Y_query'][n] = meta_dict['Y_query'][n]

    for n in range(audios_numval):
        hf['X_index'][n] = X_index[n]
        hf['Y_index'][n] = meta_dict['Y_index'][n]


print('Write hdf5 to {}'.format(packed_hdf5_path))
print("--- %s seconds training data ---" % (time.time() - start_time))
