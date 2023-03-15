import os
import time

import dcase_util
import h5py
import librosa


from myLiB.utils import *

scene_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

print("Extração de wav ")
path_meta = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\meta.csv"
path_test = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_test.csv"
path_train = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_train.csv"
path_evaluation = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_evaluate.csv"
datasetPath="..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development"


output_dir=""
fs_orig=44100
fs_new=44100
#fs_new=22050
#fs_new=16000
#fs_new=8000
divisor=15

################################################# META.CSV #############################################################
audios_dir = pd.read_csv(path_meta, sep='\t', index_col=False)
audio_names = audios_dir['filename']
audio_names_List = audios_dir['filename'].tolist() #audio_names em formato list
audio_labels = audios_dir['scene_label']
identifier = audios_dir['identifier']
source_label = audios_dir['source_label']
################################################# Train.CSV #############################################################
train_file = pd.read_csv(path_train, sep='\t')
train_filename = np.array(train_file['filename'])
train_scene_label=train_file['scene_label']

train_source_label=[]
train_indentifier=[]
start_time = time.time()
for item_id, audio_filename in enumerate(train_filename):
    try:
        idx=audio_names_List.index(audio_filename)  #procurar audio de train no meta.csv para saber source,identifier ...
        train_source_label.append(source_label[idx])
        train_indentifier.append(identifier[idx])
        # train_scene_label2.append(audio_labels[idx])
    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))
#check equality 2 list
# (train_scene_label2==train_scene_label).all()

data = np.array([train_filename,train_scene_label, train_source_label,train_indentifier]).T
df = pd.DataFrame(data, columns=['filename','scene_label', 'source_label','identifier'])


import random
airport=(df['scene_label']=='airport')
airport=random.sample((np.where(airport==True)[0]).tolist(), len(airport)//divisor)
# airport=(np.where(airport==True)[0])[0:len(airport)//divisor]

bus=(df['scene_label']=='bus')
bus=random.sample((np.where(bus==True)[0]).tolist(), len(bus)//divisor)
# bus=(np.where(bus==True)[0])[0:len(bus)//divisor]

metro=(df['scene_label']=='metro')
metro=random.sample((np.where(metro==True)[0]).tolist(), len(metro)//divisor)
# metro=(np.where(metro==True)[0])[0:len(metro)//divisor]

metro_station=(df['scene_label']=='metro_station')
metro_station=random.sample((np.where(metro_station==True)[0]).tolist(), len(metro_station)//divisor)
# metro_station=(np.where(metro_station==True)[0])[0:len(metro_station)//divisor]

park=(df['scene_label']=='park')
park=random.sample((np.where(park==True)[0]).tolist(), len(park)//divisor)
# park=(np.where(park==True)[0])[0:len(park)//divisor]

public_square=(df['scene_label']=='public_square')
public_square=random.sample((np.where(public_square==True)[0]).tolist(), len(public_square)//divisor)
# public_square=(np.where(public_square==True)[0])[0:len(public_square)//divisor]

shopping_mall=(df['scene_label']=='shopping_mall')
shopping_mall=random.sample((np.where(shopping_mall==True)[0]).tolist(), len(shopping_mall)//divisor)
# shopping_mall=(np.where(shopping_mall==True)[0])[0:len(shopping_mall)//divisor]

street_pedestrian=(df['scene_label']=='street_pedestrian')
street_pedestrian=random.sample((np.where(street_pedestrian==True)[0]).tolist(), len(street_pedestrian)//divisor)
# street_pedestrian=(np.where(street_pedestrian==True)[0])[0:len(street_pedestrian)//divisor]
street_traffic=(df['scene_label']=='street_traffic')
street_traffic=random.sample((np.where(street_traffic==True)[0]).tolist(), len(street_traffic)//divisor)
# street_traffic=(np.where(street_traffic==True)[0])[0:len(street_traffic)//divisor]

tram=(df['scene_label']=='tram')
tram=random.sample((np.where(tram==True)[0]).tolist(), len(tram)//divisor)
# tram=(np.where(tram==True)[0])[0:len(tram)//divisor]

start_time = time.time()
newData=[]
newData_scene_label=[]
newData_source_label=[]
newData_identifier=[]
for i in range(0,len(df['filename'])):
        #melhorar este if zé do pipo
    if((i in airport) or (i in bus) or (i in metro)
            or (i in metro_station) or (i in park) or (i in public_square)
            or (i in shopping_mall) or (i in street_pedestrian)
            or (i in street_traffic) or (i in tram)):
        audio = dcase_util.containers.AudioContainer().load(filename=datasetPath + "\\" + df['filename'][i], mono=True, fs=fs_orig)
        audio = audio.data
        if(fs_orig!=fs_new):
            audio = librosa.resample(audio, orig_sr=fs_orig, target_sr=fs_new)
        audio = np.float32(audio)
        newData.append(audio)
        newData_scene_label.append(df['scene_label'][i])
        newData_source_label.append(df['source_label'][i])
        newData_identifier.append(df['identifier'][i])


data2 = np.array([newData,newData_scene_label, newData_source_label,newData_identifier]).T
df2 = pd.DataFrame(data2, columns=['spec','scene_label', 'source_label','identifier'])

print(str(df2.groupby('scene_label')['source_label'].value_counts().to_string()))
print(df2['source_label'].value_counts().to_string())
print(df2['scene_label'].value_counts().to_string())
# print(df2['identifier'].value_counts().to_string())

####################################################################################
#Split data 70% Train 30%Validação
from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(newData, newData_scene_label, test_size=0.30,random_state=1)
del newData,newData_scene_label,newData_identifier,newData_source_label

####save
packed_hdf5_path = os.path.join(output_dir,'wave_' + str(fs_new)+"_D_"+str(divisor) + '.h5')
meta_dict = {
    'X_train': np.array(X_train),
    'X_validation': np.array(X_validation),
    'Y_train': np.array(Y_train),
    'Y_validation': np.array(Y_validation),
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
        shape=(len(X_train), X_train[0].shape[0]),
        dtype=np.float32)

    hf.create_dataset(
        name='X_validation',
        shape=(len(X_validation), X_validation[0].shape[0]),
        dtype=np.float32)

    for n in range(audios_numtrain):
        hf['X_train'][n] = X_train[n]
        hf['Y_train'][n] = meta_dict['Y_train'][n].encode()

    for n in range(audios_numval):
        hf['X_validation'][n] = X_validation[n]
        hf['Y_validation'][n] = meta_dict['Y_validation'][n].encode()


print('Write hdf5 to {}'.format(packed_hdf5_path))
print("--- %s seconds training data ---" % (time.time() - start_time))