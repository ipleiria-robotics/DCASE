import os
import gc
import random
import time
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_ENABLE_ONEDNN_OPTS']="1"
import h5py
import math
from pathlib import Path
import numpy as np
from tabulate import tabulate


import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import accuracy_score
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # Configure the GPU devices
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

from myLiB.funcoesPipline import *
from myLiB.models import *
from myLiB.plots import *
from myLiB.utils import *
from myLiB import tcl

try:
    import tensorflow_addons as tfa  # main package
except ModuleNotFoundError:
    #%pip install tensorflow-addons
    import tensorflow_addons as tfa

# install TF similarity if needed
try:
    import tensorflow_similarity as tfsim  # main package
except ModuleNotFoundError:
   #%pip install tensorflow_similarity
    import tensorflow_similarity as tfsim
import tensorflow_similarity.visualization as tfsim_visualization
import tensorflow_similarity.callbacks as tfsim_callbacks
import tensorflow_similarity.augmenters as tfsim_augmenters
import tensorflow_similarity.losses as tfsim_losses
import tensorflow_similarity.architectures as tfsim_architectures

import tensorflow_datasets as tfds
from PIL import Image


#tfsim.utils.tf_cap_memory()  # Avoid GPU memory blow up

# Clear out any old model state.
gc.collect()
tf.keras.backend.clear_session()

print("TensorFlow:", tf.__version__)
print("TensorFlow Similarity", tfsim.__version__)

matplotlib.use('Agg')
# matplotlib.use('TKAgg')
# Create timer instance
timer = dcase_util.utils.Timer()

ModelName = "Model" 
Path_name = "MAIO_Teste_Without_SSL_100Epocas"
DATA_PATH = Path(Path_name)
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)

image_histortResult = Path_name + "/" + "plot_history.png"   

current_dir=os.getcwd()
dcase_util.utils.setup_logging(logging_file=current_dir + "/" + Path_name +"/Results.log") 
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2023 / Task1A -- Self Supervise Learning ')                             #--------------------> ALTERAR 
log.line()

# -------------------------------------------------------------------------------------------

hdf5_path_Train = current_dir+"/"+"dcase_2024/Files/Train_fs8000_140_2048_0.04_0.02.h5"
hdf5_path_Test = current_dir+"/"+"dcase_2024/Files/Test_fs8000_140_2048_0.04_0.02.h5"
hdf5_path_QueryIndex = current_dir+"/"+"dcase_2024/Files/QueryIndex_fs8000_140_2048_0.04_0.02.h5"


def readData(hdf5_path_Train,hdf5_path_Test):
    with h5py.File(hdf5_path_Train, 'r') as hf:
        print(hf.keys())
        X_train = np.array(hf['X_train'], dtype=np.float32)
        x_val = np.array(hf['X_validation'], dtype=np.float32)
        Y_val = [x for x in hf['Y_validation']]
        Y_train = [x for x in hf['Y_train']]
        descricao = [x.decode() for x in hf['descricao']]
        descricao = "".join([str(elem) for elem in descricao])
    with h5py.File(hdf5_path_Test, 'r') as hf:
        print(hf.keys())

    return X_train,x_val,Y_val,Y_train

x_train, x_val, y_val, y_train = readData(hdf5_path_Train, hdf5_path_Test)

with h5py.File(hdf5_path_Test, 'r') as hf:
    print(hf.keys())
    x_test = np.array(hf['features'], dtype=np.float32)
    y_test = [x for x in hf['scene_label']]

with h5py.File(hdf5_path_QueryIndex, 'r') as hf:
    print(hf.keys())
    x_query = np.array(hf['X_query'], dtype=np.float32)
    y_query = [x for x in hf['Y_query']]
    x_index = np.array(hf['X_index'], dtype=np.float32)
    y_index = [x for x in hf['Y_index']]





# Shuffle Train 
from sklearn.utils import shuffle
# Assuming X_train and Y_train are your training data and labels
X_shuffled, Y_shuffled = shuffle(x_train, y_train, random_state=0)
del x_train, y_train
x_train = X_shuffled
y_train = Y_shuffled
del X_shuffled, Y_shuffled

X_shuffled, Y_shuffled = shuffle(x_val, y_val, random_state=0)
del x_val, y_val
x_val = X_shuffled
y_val = Y_shuffled
del X_shuffled, Y_shuffled

X_shuffled, Y_shuffled = shuffle(x_test, y_test, random_state=0)
del x_test, y_test
x_test = X_shuffled
y_test = Y_shuffled
del X_shuffled, Y_shuffled

# Assuming X_train and Y_train are your training data and labels
X_shuffled, Y_shuffled = shuffle(x_query, y_query, random_state=0)
x_query = X_shuffled
y_query = Y_shuffled
del X_shuffled, Y_shuffled


# Assuming X_train and Y_train are your training data and labels
X_shuffled, Y_shuffled = shuffle(x_index, y_index, random_state=0)
x_index = X_shuffled
y_index = Y_shuffled
del X_shuffled, Y_shuffled

# Convert convert_to_tensor
y_train = tf.convert_to_tensor(np.array(y_train))
y_val = tf.convert_to_tensor(np.array(y_val))
y_test = tf.convert_to_tensor(np.array(y_test))
y_query = tf.convert_to_tensor(np.array(y_query))
y_index = tf.convert_to_tensor(np.array(y_index))


# Converte para 3 canais 
input_data = x_train 
reshaped_X_train = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((x_train.shape[1], x_train.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_X_train[i, :, :, :] = reshaped_sample


# Converte para 3 canais 
input_data = x_val 
reshaped_X_validation = np.zeros((x_val.shape[0], x_val.shape[1], x_val.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((x_val.shape[1], x_val.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_X_validation[i, :, :, :] = reshaped_sample


# Converte para 3 canais 
input_data = x_test 
reshaped_X_Test = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((x_test.shape[1], x_test.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_X_Test[i, :, :, :] = reshaped_sample

# Converte para 3 canais 
input_data = x_query 
reshaped_x_query = np.zeros((x_query.shape[0], x_query.shape[1], x_query.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((x_query.shape[1], x_query.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_x_query[i, :, :, :] = reshaped_sample


# Converte para 3 canais 
input_data = x_index 
reshaped_x_index = np.zeros((x_index.shape[0], x_index.shape[1], x_index.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((x_index.shape[1], x_index.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_x_index[i, :, :, :] = reshaped_sample


del x_train, x_val, x_test, x_query, x_index
x_train = reshaped_X_train
x_val = reshaped_X_validation
x_test = reshaped_X_Test
x_query = reshaped_x_query
x_index = reshaped_x_index


x_train = tf.convert_to_tensor(x_train)
x_val = tf.convert_to_tensor(x_val)
x_test = tf.convert_to_tensor(x_test)
x_query = tf.convert_to_tensor(x_query)
x_index = tf.convert_to_tensor(x_index)


info_dataSet = tabulate(
        [
            ["train", x_train.shape, y_train.shape],
            ["Validation", x_val.shape, y_val.shape],
            ["Test", x_test.shape, y_test.shape],
            ["query", x_query.shape, y_query.shape],
            ["index", x_index.shape, y_index.shape],
        ],
        headers=["Examples", "Labels"],
)

print(info_dataSet)
log.info(info_dataSet)

# -----------------------------------------------------------------------------------------

ALGORITHM = "simclr"  # @param ["barlow", "simsiam", "simclr", "vicreg"]

# Training Parameter Setup
IMG_SIZE_X = 140
IMG_SIZE_Y = 8
IMG_SIZE_D = 3 



NUMERO_CLASSES = 10
BATCH_SIZE = 32
PRE_TRAIN_EPOCHS = 60
PRE_TRAIN_STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE
VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 5e-4
DIM = 2048  # The layer size for the projector and predictor models.
WARMUP_LR = 0.0
WARMUP_STEPS = 0
TEMPERATURE = None

if ALGORITHM == "simsiam":
    INIT_LR = 3e-2 * int(BATCH_SIZE / 256)
elif ALGORITHM == "barlow":
    INIT_LR = 1e-3  # Initial LR for the learning rate schedule.
    WARMUP_STEPS = 1000
elif ALGORITHM == "simclr":
    INIT_LR = 1e-3  # Initial LR for the learning rate schedule, see section B.1 in the paper.
    TEMPERATURE = 0.5  # Tuned for CIFAR10, see section B.9 in the paper.
elif ALGORITHM == "vicreg":
    INIT_LR = 1e-3

def get_backbone(_IMG_SIZE_X,_IMG_SIZE_Y,_IMG_SIZE_D, activation="relu", preproc_mode="torch"):
    input_shape = (_IMG_SIZE_X, _IMG_SIZE_Y, _IMG_SIZE_D)

    backbone = tfsim_architectures.ResNet18Sim(
        input_shape,
        include_top=False,  # Take the pooling layer as the output.
        pooling="avg",
    )

    #backbone = BM2(input_shape); 

    return backbone


backbone = get_backbone(IMG_SIZE_X,IMG_SIZE_Y,IMG_SIZE_D)
backbone.summary()


def img_scaling(img):
    return tf.keras.applications.imagenet_utils.preprocess_input(img, data_format=None, mode="torch")





# ------------------------------------------------------------------------------------------------------

# This final section trains two different classifiers. 
# 1-> No Pre-training: Uses a ResNet18 model and a simple linear layer.
# 2-> Pre-trained Uses the frozen pre-trained backbone from the ContrastiveModel and only trains the weights in the linear layer.

TEST_EPOCHS = 10
TEST_STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

@tf.function
def eval_augmenter(img):
    # random resize and crop. Increase the size before we crop.
    #img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
    #    img, IMG_SIZE_X, IMG_SIZE_Y, area_range=(0.4, 0.5)
    #)

    # random horizontal flip
    #img = tf.image.random_flip_left_right(img)
    #img = tf.clip_by_value(img, 0.0, 255.0)

    return img



eval_train_ds = tf.data.Dataset.from_tensor_slices((x_test, tf.keras.utils.to_categorical(y_test, NUMERO_CLASSES)))
eval_train_ds = eval_train_ds.repeat()
eval_train_ds = eval_train_ds.shuffle(1024)
eval_train_ds = eval_train_ds.map(lambda x, y: (eval_augmenter(x), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.map(lambda x, y: (img_scaling(x), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.batch(BATCH_SIZE)
eval_train_ds = eval_train_ds.prefetch(tf.data.AUTOTUNE)

eval_val_ds = tf.data.Dataset.from_tensor_slices((x_query, tf.keras.utils.to_categorical(y_query, NUMERO_CLASSES)))
eval_val_ds = eval_val_ds.repeat()
eval_val_ds = eval_val_ds.shuffle(1024)
eval_val_ds = eval_val_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
eval_val_ds = eval_val_ds.batch(BATCH_SIZE)
eval_val_ds = eval_val_ds.prefetch(tf.data.AUTOTUNE)

eval_test_ds = tf.data.Dataset.from_tensor_slices((x_index, tf.keras.utils.to_categorical(y_index, NUMERO_CLASSES)))
eval_test_ds = eval_test_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
eval_test_ds = eval_test_ds.batch(BATCH_SIZE)
eval_test_ds = eval_test_ds.prefetch(tf.data.AUTOTUNE)


def get_eval_model(_IMG_SIZE_X,_IMG_SIZE_Y,_IMG_SIZE_D, backbone, total_steps, trainable=True, lr=1.8):
    backbone.trainable = trainable
    inputs = tf.keras.layers.Input((_IMG_SIZE_X, _IMG_SIZE_Y, _IMG_SIZE_D), name="eval_input")
    x = backbone(inputs, training=trainable)
    o = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, o)
    cosine_decayed_lr = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=total_steps)
    opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
    return model

no_pt_eval_model = get_eval_model(
    _IMG_SIZE_X=IMG_SIZE_X,
    _IMG_SIZE_Y = IMG_SIZE_Y,
    _IMG_SIZE_D = IMG_SIZE_D,
    backbone=get_backbone(IMG_SIZE_X, IMG_SIZE_Y,IMG_SIZE_D, DIM),
    total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
    trainable=True,
    lr=1e-3,
)


no_pt_history = no_pt_eval_model.fit(
    eval_train_ds,
    batch_size=BATCH_SIZE,
    epochs=TEST_EPOCHS,
    steps_per_epoch=TEST_STEPS_PER_EPOCH,
    validation_data=eval_val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
)




# Comparison
no_pretrain = no_pt_eval_model.evaluate(eval_test_ds)
print("no pretrain", no_pretrain)
log.info("no pretrain    " + str(no_pretrain[0]) + "   " + str(no_pretrain[1]))

