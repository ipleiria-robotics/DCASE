import os
import time
from concurrent.futures import ThreadPoolExecutor

import dcase_util
import h5py
import librosa

from myLiB.utils import *

scene_labels = ['airport',
                'bus',
                'metro',
                'metro_station',
                'park',
                'public_square',
                'shopping_mall',
                'street_pedestrian',
                'street_traffic',
                'tram']


#TODO: ficheiro responsavel pelo pipline de extração de features/data_augmentation e armazenamento de variaveis no h5py prontas a treinar
print("Extração de features")
path_meta = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\meta.csv"
path_test = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_test.csv"
path_train = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_train.csv"
path_evaluation = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_evaluate.csv"
datasetPath="..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development"


output_dir=""

#Baseline Settings (44.1kHz) (40mels) (0.04s win) (0.02s hop) | SHAPE(40,51)
fs_Orig=44100
dicc = {'spectrogram_type': 'magnitude',
        'hop_length_seconds': 0.02,
        'win_length_seconds': 0.04,
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

fs_new=44100
#fs_new=22050
#fs_new=16000
#fs_new=8000
dicc['fs']=fs_new
dicc['fmin']=0
dicc['fmax']= fs_new // 2
####################################################
# #kerasTunerConfigs FS=8kHz [260x8] run(1)
# fs_new=8000
# dicc['fs']=fs_new
# dicc['fmin']=0
# dicc['fmax']= fs_new // 2
# dicc['n_fft']=2048
# dicc['n_mels']=260
# dicc['win_length_seconds']=0.256 #2048/fs_new
# dicc['hop_length_seconds']=0.128 #1024/fs_new

# #kerasTunerConfigs FS=8kHz [140x8] run(2)
# fs_new=8000
# dicc['fs']=fs_new
# dicc['fmin']=0
# dicc['fmax']= fs_new // 2
# dicc['n_fft']=2048
# dicc['n_mels']=140
# dicc['win_length_seconds']=0.256 #2048/fs_new
# dicc['hop_length_seconds']=0.128 #1024/fs_new

####################################################

extractor = dcase_util.features.MelExtractor(**dicc)

trainFile='Train_fs' + str(fs_new) + "_" + str(dicc['n_mels']) + "_" + str(dicc['n_fft']) + "_" + str(dicc['win_length_seconds']) + "_" + str(dicc['hop_length_seconds']) + ".h5"
testFileName='Test_fs' + str(fs_new) + "_" + str(dicc['n_mels']) + "_" + str(dicc['n_fft']) + "_" + str(dicc['win_length_seconds']) + "_" + str(dicc['hop_length_seconds']) + ".h5"
descricao="\nfs" + str(fs_new) + " n_fft_" + str(dicc['n_fft']) + " n_mels_" + str(dicc['n_mels']) \
          +"\nwin_length_" + str(dicc['win_length_seconds']) +" hop_length_" + str(dicc['hop_length_seconds']) +\
          "\nDaugAudio_" + str(dataAugmentationAudio) +"\nDaugSpec_" + str(dataAugmentationSpec)+"\nNormalize_" + str(normalizar)

# X_train = librosa.feature.delta(X_train, order=1)
# X_train = librosa.feature.delta(X_train, order=2)

if(normalizar):
    normalizer = dcase_util.data.Normalizer()

################################################# META.CSV #############################################################
audios_dir = pd.read_csv(path_meta, sep='\t', index_col=False)
audio_names = audios_dir['filename']
audio_names_List = audios_dir['filename'].tolist() #audio_names em formato list
audio_labels = audios_dir['scene_label']
newData_identifier = audios_dir['identifier']
source_label = audios_dir['source_label']
########################################################################################################################
################################################# Train.CSV #############################################################
train_file = pd.read_csv(path_train, sep='\t')
train_filename = np.array(train_file['filename'])
train_scene_label=train_file['scene_label']

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
#check equality 2 list
# (train_scene_label2==train_scene_label).all()

data = np.array([train_filename,train_scene_label, train_source_label,train_indentifier]).T
df = pd.DataFrame(data, columns=['filename','scene_label', 'source_label','identifier'])
df['Na']=(df['source_label']=='a')  #Encontrar audios do tipo A depois ignorados no dataAugmetation
del train_filename,train_scene_label,train_source_label,train_indentifier,data,source_label  #libertar espaço RAM

from audiomentations import Compose, TimeStretch, PitchShift, Shift,TimeMask
# Augment/transform/perturb the audio data
augment = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_fraction=-0.25, max_fraction=0.25, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.15, p=0.5),
    TimeMask(min_band_part=0.05, max_band_part=0.1, p=0.5),

])
augment1 = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.25),
    Shift(min_fraction=-0.25, max_fraction=0.25, p=0.25),
    TimeStretch(min_rate=0.9, max_rate=1.15, p=0.25),
    TimeMask(min_band_part=0.05, max_band_part=0.1, p=0.25),
])

from audiomentations import SpecCompose, SpecFrequencyMask
SpecAugment = SpecCompose(
    [
        SpecFrequencyMask(min_mask_fraction=0.03,max_mask_fraction=0.10,fill_mode="constant", p=0.25),
        SpecFrequencyMask(min_mask_fraction=0.03, max_mask_fraction=0.10, fill_mode="mean", p=0.25),
    ]
)

if (normalizar):
    for i in range(0,len(df['filename'])):
        audio = dcase_util.containers.AudioContainer().load(filename=datasetPath + "\\" + df['filename'][i], mono=True, fs=fs_Orig)
        audio=audio.data
        if (fs_Orig != fs_new):
            audio = librosa.resample(audio, orig_sr=fs_Orig, target_sr=fs_new)
        audio = np.float32(audio)
        spec = dcase_util.containers.FeatureContainer(data=extractor.extract(audio))
        spec=spec.data
        normalizer.accumulate(data=spec)
    # Finalize and save
    normalizer1 = normalizer.finalize().save("normalizer.pkl")

start_time = time.time()
newData=[]
newData_scene_label=[]
newData_source_label=[]
newData_identifier=[]
with ThreadPoolExecutor() as executor:
    for i in range(0,len(df['filename'])):
        audio = dcase_util.containers.AudioContainer().load(filename=datasetPath + "\\" + df['filename'][i], mono=True, fs=fs_Orig)
        audio=audio.data
        if (fs_Orig != fs_new):
            audio = librosa.resample(audio, orig_sr=fs_Orig, target_sr=fs_new)
        audio = np.float32(audio)
        spec = dcase_util.containers.FeatureContainer(data=extractor.extract(audio))
        spec=spec.data
        if (normalizar):
            spec=normalizer1.normalize(spec)
        newData.append(spec)
        newData_scene_label.append(df['scene_label'][i])
        newData_source_label.append(df['source_label'][i])
        newData_identifier.append(df['identifier'][i])

        if(df['Na'][i]== False):
            if(dataAugmentationAudio):
                audioAugment= augment(samples=audio, sample_rate=fs_new)
                spec1 = dcase_util.containers.FeatureContainer(data=extractor.extract(audioAugment))
                spec1 = spec1.data
                if (normalizar):
                    spec1 = normalizer1.normalize(spec1)
                newData.append(spec1)
                newData_scene_label.append(df['scene_label'][i])
                newData_source_label.append(df['source_label'][i])
                newData_identifier.append(df['identifier'][i])

                audioAugment1 = augment1(samples=audio, sample_rate=fs_new)
                spec2 = dcase_util.containers.FeatureContainer(data=extractor.extract(audioAugment1))
                spec2 = spec2.data
                if (normalizar):
                    spec2 = normalizer1.normalize(spec2)
                newData.append(spec2)
                newData_scene_label.append(df['scene_label'][i])
                newData_source_label.append(df['source_label'][i])
                newData_identifier.append(df['identifier'][i])

                if (dataAugmentationSpec):
                    augmented_spectrogram = SpecAugment(spec)
                    newData.append(augmented_spectrogram)
                    newData_scene_label.append(df['scene_label'][i])
                    newData_source_label.append(df['source_label'][i])
                    newData_identifier.append(df['identifier'][i])

                    augmented_spectrogram1 = SpecAugment(spec1)
                    newData.append(augmented_spectrogram1)
                    newData_scene_label.append(df['scene_label'][i])
                    newData_source_label.append(df['source_label'][i])
                    newData_identifier.append(df['identifier'][i])

                    augmented_spectrogram2 = SpecAugment(spec2)
                    newData.append(augmented_spectrogram2)
                    newData_scene_label.append(df['scene_label'][i])
                    newData_source_label.append(df['source_label'][i])
                    newData_identifier.append(df['identifier'][i])

            elif (dataAugmentationSpec):
                augmented_spectrogram = SpecAugment(spec)
                newData.append(augmented_spectrogram)
                newData_scene_label.append(df['scene_label'][i])
                newData_source_label.append(df['source_label'][i])
                newData_identifier.append(df['identifier'][i])


# for i in range(0,len(newData)):
#     augmented_spectrogram = SpecAugment(newData[i])
#     fig, axs = plt.subplots(2, dpi=150)
#     img = librosa.display.specshow(newData[i], ax=axs[0])
#     plt.colorbar(img, ax=axs[0], format='%+2.0f dB')
#     img = librosa.display.specshow(augmented_spectrogram, ax=axs[1])
#     plt.colorbar(img, ax=axs[1], format='%+2.0f dB')
#     plt.savefig('augmented_spectrogram.png')

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

#Codigo para Ver info audio gerado
# data2 = np.array([newData,newData_scene_label, newData_source_label,newData_identifier]).T
# df2 = pd.DataFrame(data2, columns=['spec','scene_label', 'source_label','identifier'])
# print(str(df2.groupby('scene_label')['source_label'].value_counts().to_string()))
# print(df2['source_label'].value_counts().to_string())
# print(df2['scene_label'].value_counts().to_string())
# # print(df2['identifier'].value_counts().to_string())

#Split data 70% Train 30%Validação
from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(newData, newData_scene_label, test_size=0.30,random_state=1, stratify=newData_identifier)
del newData,newData_scene_label,newData_identifier,newData_source_label

packed_hdf5_path = os.path.join(output_dir,trainFile)
meta_dict = {
    'X_train': np.array(X_train),
    'X_validation': np.array(X_validation),
    'Y_train': np.array(Y_train),
    'Y_validation': np.array(Y_validation),
    'descricao': np.array(descricao),
}
audios_numtrain = len(meta_dict['Y_train'])
audios_numval = len(meta_dict['Y_validation'])
with h5py.File(packed_hdf5_path, 'w') as hf:
    hf.create_dataset(
        name='Y_train',
        shape=(audios_numtrain,),
        dtype='S20')

    hf.create_dataset(
        name='Y_validation',
        shape=(audios_numval,),
        dtype='S20')

    hf.create_dataset(
        name='X_train',
        shape=(len(X_train), X_train[0].shape[0],X_train[0].shape[1]),
        dtype=np.float32)

    hf.create_dataset(
        name='X_validation',
        shape=(len(X_validation), X_validation[0].shape[0],X_validation[0].shape[1]),
        dtype=np.float32)

    hf.create_dataset(
        name='descricao',
        shape=(len(descricao.encode('utf-8')),),
        dtype='S120')

    for n in range(len(descricao.encode('utf-8'))):
        hf['descricao'][n] = descricao[n].encode('utf-8')

    for n in range(audios_numtrain):
        hf['X_train'][n] = X_train[n]
        hf['Y_train'][n] = meta_dict['Y_train'][n].encode()

    for n in range(audios_numval):
        hf['X_validation'][n] = X_validation[n]
        hf['Y_validation'][n] = meta_dict['Y_validation'][n].encode()


print('Write hdf5 to {}'.format(packed_hdf5_path))
print("--- %s seconds training data ---" % (time.time() - start_time))
del X_train, X_validation, Y_train, Y_validation,meta_dict

########################################################################################################################
################################################# Test.CSV #############################################################
if not os.path.isfile("testData.h5"):
    test_file = pd.read_csv(path_test, sep='\t')
    test_filename = np.array(test_file['filename'])
    test_scene_label=[]

    start_time = time.time()
    for item_id, audio_filename in enumerate(test_filename):
        try:
            idx=audio_names_List.index(audio_filename)  #procurar audio de test no meta.csv para saber source,identifier ...
            test_scene_label.append(audio_labels[idx])
        except:
            pass
    print("--- %s seconds ---" % (time.time() - start_time))

    data = np.array([test_filename,test_scene_label]).T
    df = pd.DataFrame(data, columns=['filename','scene_label'])

    start_time = time.time()
    testData=[]
    testData_scene_label=[]
    testData_filename=[]
    for i in range(0,len(df['filename'])):
        audio = dcase_util.containers.AudioContainer().load(filename=datasetPath + "\\" + df['filename'][i], mono=True,fs=fs_Orig)
        audio = audio.data
        if (fs_Orig != fs_new):
            audio = librosa.resample(audio, orig_sr=fs_Orig, target_sr=fs_new)
        audio = np.float32(audio)
        spec = dcase_util.containers.FeatureContainer(data=extractor.extract(audio))
        spec=spec.data
        if (normalizar):
            spec = normalizer1.normalize(spec)
        testData.append(spec)
        testData_scene_label.append(df['scene_label'][i])
        testData_filename.append(df['filename'][i])


    packed_hdf5_path = os.path.join(output_dir,testFileName)
    meta_dict = {
        'filename': np.array(testData_filename),
        'scene_label': np.array(testData_scene_label),
        'descricao': np.array(descricao),
    }
    audios_num = len(meta_dict['scene_label'])
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='filename',
            shape=(audios_num,),
            dtype='S50')  # The names are pretty long
        hf.create_dataset(
            name='scene_label',
            shape=(audios_num,),
            dtype='S20')

        hf.create_dataset(
            name='features',
            shape=(len(testData), testData[0].shape[0],testData[0].shape[1]),
            dtype=np.float32)
            #dtype=np.int16)

        hf.create_dataset(
            name='descricao',
            shape=(len(descricao.encode('utf-8')),),
            dtype='S120')

        for n in range(len(descricao.encode('utf-8'))):
            hf['descricao'][n] = descricao[n].encode('utf-8')

        for n in range(audios_num):
            #features = float32_to_int16(features_all[n])
            hf['filename'][n] = meta_dict['filename'][n].split("/")[1].encode()
            hf['scene_label'][n] = meta_dict['scene_label'][n].encode()
            hf['features'][n] = testData[n]
    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print("--- %s seconds testData ---" % (time.time() - start_time))
########################################################################################################################
########################################################################################################################