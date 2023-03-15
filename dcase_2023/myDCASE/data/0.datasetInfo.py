import os
import time

import dcase_util
from myLiB.funcoesPipline import *
from myLiB.utils import *

# Setup logging
path_meta = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\meta.csv"
path_test = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_test.csv"
path_train = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_train.csv"
path_evaluation = "..\\dcase2022_task1_baseline-main\\dataset\\TAU-urban-acoustic-scenes-2022-mobile-development\\evaluation_setup\\fold1_evaluate.csv"

# Setup logging
dcase_util.utils.setup_logging(
    logging_file=os.path.join('dataInfo.log')
)
# Get logging interface
log = dcase_util.ui.ui.FancyLogger()
log.info("DCASE-2022 dataset info")
log.line()

######################################## META.CSV INFO
audios_dir = pd.read_csv(path_meta, sep='\t', index_col=False)
audio_names = audios_dir['filename']
audio_names_List = audios_dir['filename'].tolist() #audio_names em formato list
audio_labels = audios_dir['scene_label']
identifier = audios_dir['identifier']
source_label = audios_dir['source_label']

log.info("META.CSV INFO")
log.info("Numero total de audios")
log.info(str(len(audio_names)))
log.line()
log.info("Numero de amostras por audio label")
log.info(audio_labels.value_counts().to_string())
log.line()
log.info("Numero de amostras por source label")
log.info(source_label.value_counts().to_string())
log.line()
data = np.array([audio_labels, source_label]).T
df = pd.DataFrame(data, columns=['scene_label', 'source_label'])
log.info("Numero de amostras de audio por source label")
log.info(str(df.groupby('scene_label')['source_label'].value_counts().to_string()))
log.line()

########################################  fold1_train

# remover os itens de avaliação pois não contam para o balance do dataset de train
train_file = pd.read_csv(path_train, sep='\t')
train_filename = np.array(train_file['filename'])
train_scene_label=train_file['scene_label']

train_source_label=[]
start_time = time.time()
for item_id, audio_filename in enumerate(train_filename):
    try:
        idx=audio_names_List.index(audio_filename)  #procurar audio de train  no meta.csv
        train_source_label.append(source_label[idx])
    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))

data = np.array([train_filename,train_scene_label, train_source_label]).T
df = pd.DataFrame(data, columns=['filename','scene_label', 'source_label'])

log.info("train_file.CSV INFO")
log.info("Numero total de audios")
log.info(str(len(train_filename)))
log.line()
log.info("Numero de amostras por audio label")
log.info(df['scene_label'].value_counts().to_string())
log.line()
log.info("Numero de amostras por source label")
log.info(df['source_label'].value_counts().to_string())
log.line()
log.info("Numero de amostras de audio por source label")
log.info(str(df.groupby('scene_label')['source_label'].value_counts().to_string()))
log.line()




# remover os itens de avaliação pois não contam para o balance do dataset de test
test_file = pd.read_csv(path_test, sep='\t')
test_filename = np.array(test_file['filename'])

test_scene_label=[]
test_source_label=[]
start_time = time.time()
for item_id, audio_filename in enumerate(test_filename):
    try:
        idx=audio_names_List.index(audio_filename)  #procurar audio de test  no meta.csv
        test_source_label.append(source_label[idx])
        test_scene_label.append(audio_labels[idx])
    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))

data = np.array([test_filename,test_scene_label, test_source_label]).T
df = pd.DataFrame(data, columns=['filename','scene_label', 'source_label'])

log.info("test_file.CSV INFO")
log.info("Numero total de audios")
log.info(str(len(test_filename)))
log.line()
log.info("Numero de amostras por audio label")
log.info(df['scene_label'].value_counts().to_string())
log.line()
log.info("Numero de amostras por source label")
log.info(df['source_label'].value_counts().to_string())
log.line()
log.info("Numero de amostras de audio por source label")
log.info(str(df.groupby('scene_label')['source_label'].value_counts().to_string()))
log.line()



