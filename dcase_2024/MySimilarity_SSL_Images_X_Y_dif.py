'''
    FONT: https://github.com/tensorflow/similarity/blob/master/examples/unsupervised_hello_world.ipynb

'''


'''
-> Tensorflow Similarity provides a set of network architectures, 
    losses, and data augmentations that are common across a number of 
    self-supervised learning techniques. 

-> View: A view represents an augmented example.    
-> Backbone: Refers to the model that learns the Representation that we will use for downstream tasks.
-> Projector: Is an MLP model that projects the backbone representation of a view to an Embedding 
   that is contrasted with other views using a specialized contrastive loss.
-> Predictor: Is an optional MLP model that is used, in conjunction with gradient stopping, 
   in some recent architectures to further improve the representation quality.
-> Stop Gradient: Is used by some algorithms to ensure that we only propagate the update from the main view and not the contrasting view.  


'''



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

# INFO messages are not printed.
# This must be run before loading other modules.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

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

current_dir=os.getcwd()

matplotlib.use('Agg')
# matplotlib.use('TKAgg')
# Create timer instance
#timer = dcase_util.utils.Timer()




ModelName = "Model"   
#Path_name = "simsiam_batch_450_epoca_600_ResNet18_DCASE_allAug"            
Path_name = "Maio_Teste_sem_eval_augmenter"                                          #--------------------> ALTERAR 
DATA_PATH = Path(Path_name)                                               #--------------------> ALTERAR 
if not DATA_PATH.exists():                                                             
    DATA_PATH.mkdir(parents=True)
image_histortResult = Path_name + "/" + "plot_history.png"                                         #--------------------> ALTERAR                                                       
dcase_util.utils.setup_logging(logging_file=current_dir + "/" + Path_name +"/Results.log")          #--------------------> ALTERAR 
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2023 / Task1A -- Self Supervise Learning ')                             #--------------------> ALTERAR 
log.line()

#######################################################################

hdf5_path_Train = current_dir+"/"+"dcase_2024/Files/Train_fs8000_140_2048_0.04_0.02.h5"
hdf5_path_Test = current_dir+"/"+"dcase_2024/Files/Test_fs8000_140_2048_0.04_0.02.h5"
hdf5_path_QueryIndex = current_dir+"/"+"dcase_2024/Files/QueryIndex_fs8000_140_2048_0.04_0.02.h5"




'''
Train: Used for self-supervised pre-training and for training the classifiers.
Test: Reserved for the classifier evaluation.
-> we are going to partition the train data into the following additional splits:
    Validation: Data used for validation metrics during the pre-training phase.
    Query and Index: Data used to compute matching metrics. The query data is used to retrieve the nearest indexed examples.

'''

def readData(hdf5_path_Train,hdf5_path_Test):
    with h5py.File(hdf5_path_Train, 'r') as hf:
        print(hf.keys())
        X_train = np.array(hf['X_train'], dtype=np.float32)
        X_validation = np.array(hf['X_validation'], dtype=np.float32)
        Y_validation = [x for x in hf['Y_validation']]
        Y_train = [x for x in hf['Y_train']]
        descricao = [x.decode() for x in hf['descricao']]
        descricao = "".join([str(elem) for elem in descricao])
    with h5py.File(hdf5_path_Test, 'r') as hf:
        print(hf.keys())

    return X_train,X_validation,Y_validation,Y_train

X_train, X_validation, Y_validation, Y_train = readData(hdf5_path_Train, hdf5_path_Test)

from sklearn.utils import shuffle

# Assuming X_train and Y_train are your training data and labels
X_shuffled, Y_shuffled = shuffle(X_train, Y_train, random_state=0)
X_train = X_shuffled
Y_train = Y_shuffled
del X_shuffled, Y_shuffled


# Assuming X_train and Y_train are your training data and labels
X_shuffled, Y_shuffled = shuffle(X_validation, Y_validation, random_state=0)
X_validation = X_shuffled
Y_validation = Y_shuffled
del X_shuffled, Y_shuffled


with h5py.File(hdf5_path_Test, 'r') as hf:
    print(hf.keys())
    X_Test = np.array(hf['features'], dtype=np.float32)
    Y_Test = [x for x in hf['scene_label']]


with h5py.File(hdf5_path_QueryIndex, 'r') as hf:
    print(hf.keys())
    x_query = np.array(hf['X_query'], dtype=np.float32)
    y_query = [x for x in hf['Y_query']]
    x_index = np.array(hf['X_index'], dtype=np.float32)
    y_index = [x for x in hf['Y_index']]


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


# Convert Y Data 
Y_train = tf.convert_to_tensor(np.array(Y_train))
Y_validation = tf.convert_to_tensor(np.array(Y_validation))
Y_Test = tf.convert_to_tensor(np.array(Y_Test))
y_query = tf.convert_to_tensor(np.array(y_query))
y_index = tf.convert_to_tensor(np.array(y_index))

########################## Converter para 3 canais ###############################################

# Converte para 3 canais 
input_data = X_train 
reshaped_X_train = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((X_train.shape[1], X_train.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_X_train[i, :, :, :] = reshaped_sample

# Converte para 3 canais 
input_data = X_validation 
reshaped_X_validation = np.zeros((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((X_validation.shape[1], X_validation.shape[2], 3))
    for j in range(3):
        reshaped_sample[:, :, j] = sample
    reshaped_X_validation[i, :, :, :] = reshaped_sample


# Converte para 3 canais 
input_data = X_Test 
reshaped_X_Test = np.zeros((X_Test.shape[0], X_Test.shape[1], X_Test.shape[2], 3), dtype=np.float32)
for i in range(input_data.shape[0]):
    sample = input_data[i, :, :]  # Extract one sample at a time
    reshaped_sample = np.zeros((X_Test.shape[1], X_Test.shape[2], 3))
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

del X_train, X_validation, X_Test, x_query, x_index
X_train = reshaped_X_train
X_validation = reshaped_X_validation
X_Test = reshaped_X_Test
x_query = reshaped_x_query
x_index = reshaped_x_index

##########################################################################################################

X_train = tf.convert_to_tensor(X_train)
X_validation = tf.convert_to_tensor(X_validation)
X_Test = tf.convert_to_tensor(X_Test)
x_query = tf.convert_to_tensor(x_query)
x_index = tf.convert_to_tensor(x_index)



#######################################################################################################

info_dataSet = tabulate(
        [
            ["train", X_train.shape, Y_train.shape],
            ["Validation", X_validation.shape, Y_validation.shape],
            ["Test", X_Test.shape, Y_Test.shape],
            ["Query", x_query.shape, y_query.shape],
            ["Index", x_index.shape, y_index.shape],
        ],
        headers=["Examples", "Labels"],
)

print(info_dataSet)
log.info(info_dataSet)


# Assuming mel_spectrogram is your mel spectrogram data
#mel_image = (X_train[0] - np.min(X_train[0])) / (np.max(X_train[0]) - np.min(X_train[0]))

# Plot the mel spectrogram
#plt.figure(figsize=(4, 4))  # Set the figure size
#plt.imshow(mel_image, aspect='auto', origin='lower')
#plt.axis('off')  # Turn off axis
#plt.tight_layout()
    
# Save the image
#image_file = f'mel_spectrogram_0.png'
#plt.savefig(image_file)
#plt.close()




# Self-Supervised Training Setup
#SimCLR: Only requires the Backbone and the projector and uses a contrastive cross-entropy loss. 
ALGORITHM = "simsiam"  # @param ["barlow", "simsiam", "simclr", "vicreg"]


# Training Parameter Setup
IMG_SIZE_X = 140
IMG_SIZE_Y = 8
BATCH_SIZE = 256  # 50 
PRE_TRAIN_EPOCHS = 40 # 800
PRE_TRAIN_STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 5e-4 # 1e-6
DIM = 2048  # The layer size for the projector and predictor models.o
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

def img_scaling(img):
    return tf.keras.applications.imagenet_utils.preprocess_input(img, data_format=None, mode="torch")




@tf.function
def simsiam_augmenter(img, blur=True, area_range=(0.2, 1.0)):

    """SimSiam augmenter.

    The SimSiam augmentations are based on the SimCLR augmentations, but have
    some important differences.
    * The crop area lower bound is 20% instead of 8%.
    * The color jitter and grayscale are applied separately instead of together.
    * The color jitter ranges are much smaller.
    * Blur is not applied for the cifar10 dataset.

    args:
        img: Single image tensor of shape (H, W, C)
        blur: If true, apply blur. Should be disabled for cifar10.
        area_range: The upper and lower bound of the random crop percentage.

    returns:
        A single image tensor of shape (H, W, C) with values between 0.0 and 1.0.
    """

    # random resize and crop. Increase the size before we crop.
    #img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
    #    img, IMG_SIZE_X, IMG_SIZE_Y, area_range=area_range
    #)

    # The following transforms expect the data to be [0, 1]
    #img /= 255.0

    # random color jitter
    #def _jitter_transform(x):
    #    return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
    #        x,
    #        np.random.uniform(0.0, 0.4),
    #        np.random.uniform(0.0, 0.4),
    #        np.random.uniform(0.0, 0.4),
    #        np.random.uniform(0.0, 0.1),
    #        "multiplicative",
    #    )

    #img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=0.8, x=img)

    # # random grayscale
    #def _grascayle_transform(x):
    #    return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

    #img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=0.2, x=img)

    # optional random gaussian blur
    #if blur:
    #    img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, p=0.5)

    # random horizontal flip
    #img = tf.image.random_flip_left_right(img)
    

    # scale the data back to [0, 255]
    #img = img * 255.0
    #img = tf.clip_by_value(img, 0.0, 255.0)


    return img


@tf.function()
def process(img):
    view1 = simsiam_augmenter(img, blur=False, area_range=(0.2, 1.0))
    view1 = img_scaling(view1)
    view2 = simsiam_augmenter(img, blur=False, area_range=(0.2, 1.0))
    view2 = img_scaling(view2)    

    return (view1, view2)



train_ds = tf.data.Dataset.from_tensor_slices(X_train)
train_ds = train_ds.repeat()
train_ds = train_ds.shuffle(1024)
train_ds = train_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(X_validation)
val_ds = val_ds.repeat()
val_ds = val_ds.shuffle(1024)
val_ds = val_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)




def get_backbone(img_sizeX,img_sizeY, activation="relu", preproc_mode="torch"):
    input_shape = (img_sizeX, img_sizeY, 3)

    backbone = tfsim_architectures.ResNet50Sim(
        input_shape,
        include_top=False,  # Take the pooling layer as the output.
        pooling="avg",
    )
    return backbone
    #return Baseline(input_shape)


backbone = get_backbone(IMG_SIZE_X, IMG_SIZE_Y)
backbone.summary(print_fn=log.info)


if(True):
    
    #  Projector Model
    projector = None  # Passing None will automatically build the default projector.
         # Uncomment to build a custom projector.
    def get_projector(input_dim, dim, activation="relu", num_layers: int = 3):
         inputs = tf.keras.layers.Input((input_dim,), name="projector_input")
         x = inputs

         for i in range(num_layers - 1):
             x = tf.keras.layers.Dense(
                 dim,
                 use_bias=False,
                 kernel_initializer=tf.keras.initializers.LecunUniform(),
                 name=f"projector_layer_{i}",
             )(x)
             x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=f"batch_normalization_{i}")(x)
             x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_{i}")(x)
         x = tf.keras.layers.Dense(
             dim,
             use_bias=False,
             kernel_initializer=tf.keras.initializers.LecunUniform(),
             name="projector_output",
         )(x)
         x = tf.keras.layers.BatchNormalization(
             epsilon=1.001e-5,
             center=False,  # Page:5, Paragraph:2 of SimSiam paper
             scale=False,  # Page:5, Paragraph:2 of SimSiam paper
             name=f"batch_normalization_ouput",
         )(x)
         # Metric Logging layer. Monitors the std of the layer activations.
         # Degnerate solutions colapse to 0 while valid solutions will move
         # towards something like 0.0220. The actual number will depend on the layer size.
         o = tfsim.layers.ActivationStdLoggingLayer(name="proj_std")(x)
         projector = tf.keras.Model(inputs, o, name="projector")
         return projector
    
    projector = get_projector(input_dim=backbone.output.shape[-1], dim=DIM, num_layers=2)
    projector.summary()
    projector = None  # Passing None will automatically build the default projector.


    # Predictor model
    # The predictor model is used by BYOL and SimSiam, and is an additional 2 layer MLP containing a bottleneck in the hidden layer.
    predictor = None  # Passing None will automatically build the default predictor.
    def get_predictor(input_dim, hidden_dim=512, activation="relu"):       
        inputs = tf.keras.layers.Input(shape=(input_dim,), name="predictor_input")
        x = inputs

        x = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name="predictor_layer_0",
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="batch_normalization_0")(x)
        x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_0")(x)

        x = tf.keras.layers.Dense(
            input_dim,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name="predictor_output",
        )(x)
         # Metric Logging layer. Monitors the std of the layer activations.
         # Degnerate solutions colapse to 0 while valid solutions will move
         # towards something like 0.0220. The actual number will depend on the layer size.
        o = tfsim.layers.ActivationStdLoggingLayer(name="pred_std")(x)
        predictor = tf.keras.Model(inputs, o, name="predictor")
        return predictor
    
    predictor = get_predictor(input_dim=DIM, hidden_dim=512)
    predictor.summary()
    predictor = None  # Passing None will automatically build the default predictor.

    # Self-Supervised Algorithms 
    contrastive_model = tfsim.models.create_contrastive_model(
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        algorithm=ALGORITHM,
        name=ALGORITHM,
    )

    if ALGORITHM == "simsiam":
        loss = tfsim_losses.SimSiamLoss(projection_type="cosine_distance", name=ALGORITHM)
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=INIT_LR,
            decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
        )
        wd_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=WEIGHT_DECAY,
            decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
        )
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_decayed_fn, weight_decay=wd_decayed_fn, momentum=0.9)
    elif ALGORITHM == "barlow":
        loss = tfsim_losses.Barlow(name=ALGORITHM)
        optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
    elif ALGORITHM == "simclr":
        loss = tfsim_losses.SimCLRLoss(name=ALGORITHM, temperature=TEMPERATURE)
        optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
    elif ALGORITHM == "vicreg":
        loss = tfsim_losses.VicReg(name=ALGORITHM)
        optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR)
    else:
        raise ValueError(f"{ALGORITHM} is not supported.")

    contrastive_model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    # Callbacks

    log_dir = DATA_PATH / "models" / "logs" / f"{loss.name}_{time.time()}"
    chkpt_dir = DATA_PATH / "models" / "checkpoints" / f"{loss.name}_{time.time()}"

    evb = tfsim_callbacks.EvalCallback(
        img_scaling(tf.cast(x_query, tf.float32)),
        y_query,
        img_scaling(tf.cast(x_index, tf.float32)),
        y_index,
        metrics=["binary_accuracy"],
        k=1,
        tb_logdir=log_dir,
    )

    tbc = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=100,
    )
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpt_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Model Training
    history = contrastive_model.fit(
        train_ds,
        epochs=PRE_TRAIN_EPOCHS,
        steps_per_epoch=PRE_TRAIN_STEPS_PER_EPOCH,
        validation_data=val_ds,
        validation_steps=VAL_STEPS_PER_EPOCH,
        callbacks=[evb, tbc, mcp],
        verbose=1,
    )


    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history["loss"])
    plt.grid()
    plt.title(f"{loss.name} - loss")
        

    plt.subplot(1, 3, 2)
    plt.plot(history.history["proj_std"], label="proj")
    if "pred_std" in history.history:
        plt.plot(history.history["pred_std"], label="pred")
    plt.grid()
    plt.title(f"{loss.name} - std metrics")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history["binary_accuracy"], label="acc")
    plt.grid()
    plt.title(f"{loss.name} - match metrics")
    plt.legend()

    plt.show()
    plt.savefig(image_histortResult)

    contrastive_model.save(DATA_PATH / "models" / ModelName)
    del contrastive_model



contrastive_model = tf.keras.models.load_model(
    DATA_PATH / "models" / ModelName,
    custom_objects={
        "ContrastiveModel": tfsim.models.ContrastiveModel,
        "ActivationStdLoggingLayer": tfsim.layers.ActivationStdLoggingLayer,
    },
)

'''
-> The original train data is partitioned into eval_train and eval_val splits 
    and a simplified augmentation is applied to the training data.

-> The models are then trained for x epochs and the classification 
    accuracy is evaluated on the held out test split.

'''

TEST_EPOCHS = 10
TEST_STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE


@tf.function
def eval_augmenter(img):
    # random resize and crop. Increase the size before we crop.
    #img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
    #    img, IMG_SIZE_X, IMG_SIZE_Y, area_range=(0.2, 1.0)
    #)
    # random horizontal flip
    #img = tf.image.random_flip_left_right(img)
    #img = tf.clip_by_value(img, 0.0, 255.0)

    return img


eval_train_ds = tf.data.Dataset.from_tensor_slices((X_train, tf.keras.utils.to_categorical(Y_train, 10)))
eval_train_ds = eval_train_ds.repeat()
eval_train_ds = eval_train_ds.shuffle(1024)
eval_train_ds = eval_train_ds.map(lambda x, y: (eval_augmenter(x), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.map(lambda x, y: (img_scaling(x), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.batch(BATCH_SIZE)
eval_train_ds = eval_train_ds.prefetch(tf.data.AUTOTUNE)

eval_val_ds = tf.data.Dataset.from_tensor_slices((X_validation, tf.keras.utils.to_categorical(Y_validation, 10)))
eval_val_ds = eval_val_ds.repeat()
eval_val_ds = eval_val_ds.shuffle(1024)
eval_val_ds = eval_val_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
eval_val_ds = eval_val_ds.batch(BATCH_SIZE)
eval_val_ds = eval_val_ds.prefetch(tf.data.AUTOTUNE)

eval_test_ds = tf.data.Dataset.from_tensor_slices((X_validation, tf.keras.utils.to_categorical(Y_validation, 10)))
eval_test_ds = eval_test_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
eval_test_ds = eval_test_ds.batch(BATCH_SIZE)
eval_test_ds = eval_test_ds.prefetch(tf.data.AUTOTUNE)

def get_eval_model(img_sizeX,img_sizeY, backbone, total_steps, trainable=True, lr=1.8):
    backbone.trainable = trainable
    inputs = tf.keras.layers.Input((img_sizeX, img_sizeY, 3), name="eval_input")
    x = backbone(inputs, training=trainable)
    o = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, o)
    cosine_decayed_lr = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=total_steps)
    opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
    return model


# No Pretrain  ->  Uses a ModelGeneric and a simple linear layer.
no_pt_eval_model = get_eval_model(
    img_sizeX=IMG_SIZE_X,
    img_sizeY=IMG_SIZE_Y,
    backbone=get_backbone(IMG_SIZE_X, IMG_SIZE_Y, DIM),
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

# Pre-trained Uses the frozen pre-trained backbone from the ContrastiveModel and only trains the weights in the linear layer.
pt_eval_model = get_eval_model(
    img_sizeX=IMG_SIZE_X,
    img_sizeY=IMG_SIZE_Y,
    backbone=contrastive_model.backbone,
    total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
    trainable=False,
    lr=30.0,
)

pt_eval_model.summary()

pt_history = pt_eval_model.fit(
    eval_train_ds,
    batch_size=BATCH_SIZE,
    epochs=TEST_EPOCHS,
    steps_per_epoch=TEST_STEPS_PER_EPOCH,
    validation_data=eval_val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
)



# Comparison
no_pretrain = no_pt_eval_model.evaluate(eval_test_ds)
print("no pretrain",no_pretrain)
log.info("no pretrain    loss: " + str(no_pretrain[0]) + "   Acc: " + str(no_pretrain[1]))

pretrained = pt_eval_model.evaluate(eval_test_ds)
print("pretrained", pretrained)
log.info("pretrained   loss: " + str(pretrained[0]) + "  Acc: " + str(pretrained[1]))


