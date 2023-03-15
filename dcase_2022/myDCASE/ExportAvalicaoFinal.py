# from NeSsi import nessi
import time

import h5py
import librosa

from myLiB.funcoesPipline import *
from myLiB.utils import *

# Create timer instance
timer = dcase_util.utils.Timer()
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


#Dados avaliaçãoFinal
path_test="data/TAU-urban-acoustic-scenes-2022-mobile-evaluation/evaluation_setup/fold1_test.csv"
datasetPath="data/TAU-urban-acoustic-scenes-2022-mobile-evaluation"

#BaseSettings
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
#dataSettings FS=8kHz [140x8]
fs_new=8000
dicc['fs']=fs_new
dicc['fmin']=0
dicc['fmax']= fs_new // 2
dicc['n_fft']=2048
dicc['n_mels']=140
dicc['win_length_seconds']=0.256 #2048/fs_new
dicc['hop_length_seconds']=0.128 #1024/fs_newkeras_model_0
extractor = dcase_util.features.MelExtractor(**dicc)

directory="avaliacao/"
fold_model_filename="modelo3_4-Sub4.tflite"
path_estimated_scene = "Sub4.csv"
if os.path.isdir(directory): pass
else: os.makedirs(directory)
testFileName="testData.h5"

dcase_util.utils.setup_logging(logging_file=os.path.join(directory+"Avaliacao_task1a_v2.log"))
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2022 / Task1A -- low-complexity Acoustic Scene Classification')
log.line()


#extrair e guardar data de avaliação
if not os.path.isfile(directory+testFileName):
    print("Extração de features")
    test_file = pd.read_csv(path_test, sep='\t')
    test_filename = np.array(test_file['filename'])
    test_scene_label=[]
    data = np.array([test_filename]).T
    df = pd.DataFrame(data, columns=['filename'])

    start_time = time.time()
    testData=[]
    testData_filename=[]
    for i in range(0,len(df['filename'])):
        audio = dcase_util.containers.AudioContainer().load(filename=datasetPath + "/" + df['filename'][i], mono=True,fs=fs_Orig)
        audio = audio.data
        if (fs_Orig != fs_new):
            audio = librosa.resample(audio, orig_sr=fs_Orig, target_sr=fs_new)
        audio = np.float32(audio)
        spec = dcase_util.containers.FeatureContainer(data=extractor.extract(audio))
        spec=spec.data
        testData.append(spec)
        testData_filename.append(df['filename'][i])

    packed_hdf5_path = os.path.join(directory,testFileName)
    meta_dict = {
        'filename': np.array(testData_filename),
    }
    audios_num = len(meta_dict['filename'])
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='filename',
            shape=(audios_num,),
            dtype='S50')  # The names are pretty long
        hf.create_dataset(
            name='features',
            shape=(len(testData), testData[0].shape[0],testData[0].shape[1]),
            dtype=np.float32)
        for n in range(audios_num):
            hf['filename'][n] = meta_dict['filename'][n].split("/")[1].encode()
            hf['features'][n] = testData[n]
    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print("--- %s seconds testData ---" % (time.time() - start_time))


if os.path.isdir(directory): pass
else: os.makedirs(directory)


# macc, params = nessi.get_model_size(directory+fold_model_filename, 'tflite')
# nessi.validate(macc, params, log)

log.section_header('testing')
timer.start()
if not os.path.isfile(directory+"/"+path_estimated_scene):
    # Loop over all cross-validation folds and learn acoustic models
    with h5py.File(directory+testFileName, 'r') as hf:
        print(hf.keys())
        test_features = np.array(hf['features'])
        test_filename = [x.decode() for x in hf['filename']]

    do_testing(scene_labels, fold_model_filename, path_estimated_scene, test_features, test_filename, log,directory)
timer.stop()
log.foot(time=timer.elapsed())
