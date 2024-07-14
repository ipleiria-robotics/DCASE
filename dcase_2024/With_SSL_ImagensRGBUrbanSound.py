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
import csv

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
#Path_name = "UrbanTeste_simsiam_Epocas100_batch_16"
Path_name = "CCCCCCCCCCCCCCCCCCCCC"

ALGORITHM = "barlow"  # @param ["barlow", "simsiam", "simclr", "vicreg"]
WITH_TRAIN_SSL = False 

IMG_SIZE_X = 51
IMG_SIZE_Y = 51
IMG_SIZE_D = 3 
NUMERO_CLASSES = 10
BATCH_SIZE = 64
PRE_TRAIN_EPOCHS = 200
VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 10e-10 #5e-4
DIM = 2048  # The layer size for the projector and predictor models.
WARMUP_LR = 0.0
WARMUP_STEPS = 0
TEMPERATURE = None

DATA_PATH = Path(Path_name)
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)

image_histortResult = Path_name + "/" + "plot_history.png"   
image_histortResult_FineTunnning  = Path_name + "/" + "plot_history_fineTunning.png"   

current_dir=os.getcwd()
dcase_util.utils.setup_logging(logging_file=current_dir + "/" + Path_name +"/Results.log") 
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2023 / Task1A -- Self Supervise Learning ')                             #--------------------> ALTERAR 
log.line()

from PIL import Image
import numpy as np

# -------------------------------------------------------------------------------------------
def simsiam_augmenterAAA(img, blur=False, area_range=(0.2, 1.0)):

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
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
        img, 52, 52, area_range=area_range
    )

    # The following transforms expect the data to be [0, 1]
    img /= 255.0

    # random color jitter
    def _jitter_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
            x,
            np.random.uniform(0, 0.4),
            np.random.uniform(0, 0.4),
            np.random.uniform(0, 0.4),
            np.random.uniform(0, 0.1),
            "multiplicative",
        )

    #img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=0.2, x=img)

    # # random grayscale
    def _grascayle_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

    #img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=0.2, x=img)

    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    

    # scale the data back to [0, 255]
    img = img * 255.0

    #img = tf.clip_by_value(img, 0.0, 255.0)


    numpy_array = img.numpy()
    numpy_array  = np.array(numpy_array, dtype=np.uint8)
    image = Image.fromarray(numpy_array)
    image.save(current_dir+"/"+ "dcase_2024/Urban_augmenter.jpg")

    return img


def readCsvImages(csv_file, roor_dir):
    X = []
    Y = []
    i = 0
    with open(csv_file) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            if True:
                img_filename = row[0]
                img_path = os.path.join(roor_dir, img_filename)
                file_name = img_filename.split('/')
                #image = Image.open(roor_dir + file_name[1] + ".png" ).convert('RGB')
                root1 = roor_dir + img_filename
                image = tf.io.read_file(root1)
                image = tf.image.decode_image(image, channels=3, dtype=tf.uint8)
                image = tf.image.resize(image,[IMG_SIZE_X,IMG_SIZE_Y])
                image = tf.cast(image,tf.uint8)

                resized_image_np = image.numpy()
                resized_image_pil = Image.fromarray(resized_image_np)
                resized_image_pil.save(current_dir+"/"+ "dcase_2024/UrbanOriginal.jpg")

                simsiam_augmenterAAA(image)

                label = row[7]
                if label == 'dog_bark':
                    label = 0
                elif label == 'jackhammer':
                    label = 1  
                elif label == 'engine_idling':
                    label = 2 
                elif label == 'children_playing':
                    label = 3 
                elif label == 'siren':
                    label = 4 
                elif label == 'street_music':
                    label = 5 
                elif label == 'gun_shot':
                    label = 6 
                elif label == 'air_conditioner':
                    label = 7              
                elif label == 'drilling':
                    label = 8 
                elif label == 'car_horn':
                    label = 9        
                Y.append(label)
                #np_image = np.array(image)
                #np_image = np_image / 255.0
                X.append(image)
                print(i)
                i = i + 1

    return X,Y


X = []
Y = []
#X, Y = readCsvImages(csv_file = current_dir+"/"+ "dcase_2024/fold1_train.csv", roor_dir = current_dir+"/"+ "dcase_2024/Archive/")
X, Y = readCsvImages(csv_file = current_dir+"/"+ "dcase_2024/UrbanSound8K.csv", roor_dir = current_dir+"/"+ "dcase_2024/fold1/")

from sklearn.model_selection import train_test_split
# First split the data into training+validation and testing sets (80% train+val, 20% test)
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Then split the training+validation set into training and validation sets (90% train, 10% val)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=42)

X_qy = []
Y_qy = []
X_qy, Y_qy = readCsvImages(csv_file = current_dir+"/"+ "dcase_2024/UrbanSound8K_val.csv", roor_dir = current_dir+"/"+ "dcase_2024/fold1_val/")

X_query, X_index, Y_query, Y_index = train_test_split(X_qy, Y_qy, test_size=0.5, random_state=42)


# Convert convert_to_tensor
y_train = tf.convert_to_tensor(Y_train)
x_train = tf.convert_to_tensor(X_train)


y_val = tf.convert_to_tensor(Y_val)
x_val = tf.convert_to_tensor(X_val)


y_test = tf.convert_to_tensor(Y_test)
x_test = tf.convert_to_tensor(X_test)


y_query = tf.convert_to_tensor(Y_query)
x_query = tf.convert_to_tensor(X_query)


y_index = tf.convert_to_tensor(Y_index)
x_index = tf.convert_to_tensor(X_index)



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
PRE_TRAIN_STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

if ALGORITHM == "simsiam":
    INIT_LR = 1e-3  * int(BATCH_SIZE / 256)
elif ALGORITHM == "barlow":
    INIT_LR = 1e-6  # Initial LR for the learning rate schedule. -> Alterado para este valor se não dispara para o infinito 
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
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
        img, IMG_SIZE_X, IMG_SIZE_Y, area_range=area_range
    )

    # The following transforms expect the data to be [0, 1]
    img /= 255.0


    # random color jitter
    def _jitter_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
            x,
            np.random.uniform(0, 0.4),
            np.random.uniform(0, 0.4),
            np.random.uniform(0, 0.4),
            np.random.uniform(0, 0.1),
            "multiplicative",
        )

    img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=0.8, x=img)

    # # random grayscale
    def _grascayle_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

    #img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=0.2, x=img)

    # optional random gaussian blur
    if blur:
        img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, p=0.5)

    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    

    # scale the data back to [0, 255]
    img = img * 255.0

    img = tf.clip_by_value(img, 0.0, 255.0)
    return img

@tf.function()
def process(img):
    view1 = simsiam_augmenter(img, blur=False, area_range=(0.2, 1.0))
    view1 = img_scaling(view1)
    view2 = simsiam_augmenter(img, blur=False, area_range=(0.2, 1.0))
    view2 = img_scaling(view2)    

    return (view1, view2)

# ------------------------------------------------------------------------------------------------------

if(WITH_TRAIN_SSL):   
    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(x_val)
    val_ds = val_ds.repeat()
    val_ds = val_ds.shuffle(1024)
    val_ds = val_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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
    #del contrastive_model


#if WITH_TRAIN_SSL == False:
#    contrastive_model = tf.keras.models.load_model(
#        DATA_PATH / "models" / ModelName,
#        custom_objects={
#            "ContrastiveModel": tfsim.models.ContrastiveModel,
#            "ActivationStdLoggingLayer": tfsim.layers.ActivationStdLoggingLayer,
#        },
#    )

    

# ------------------------------------------------------------------------------------------------------

# This final section trains two different classifiers. 
# 1-> No Pre-training: Uses a ResNet18 model and a simple linear layer.
# 2-> Pre-trained Uses the frozen pre-trained backbone from the ContrastiveModel and only trains the weights in the linear layer.

TEST_EPOCHS = 100
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


#eval_train_ds = tf.data.Dataset.from_tensor_slices((x_train, tf.keras.utils.to_categorical(y_train, NUMERO_CLASSES)))
eval_train_ds = tf.data.Dataset.from_tensor_slices((x_test, tf.keras.utils.to_categorical(y_test, NUMERO_CLASSES)))
eval_train_ds = eval_train_ds.repeat()
eval_train_ds = eval_train_ds.shuffle(1024)
eval_train_ds = eval_train_ds.map(lambda x, y: (eval_augmenter(x), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.map(lambda x, y: (img_scaling(tf.cast(x, tf.float32)), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.batch(BATCH_SIZE)
eval_train_ds = eval_train_ds.prefetch(tf.data.AUTOTUNE)

#eval_val_ds = tf.data.Dataset.from_tensor_slices((x_val, tf.keras.utils.to_categorical(y_val, NUMERO_CLASSES)))
eval_val_ds = tf.data.Dataset.from_tensor_slices((x_query, tf.keras.utils.to_categorical(y_query, NUMERO_CLASSES)))
eval_val_ds = eval_val_ds.repeat()
eval_val_ds = eval_val_ds.shuffle(1024)
eval_val_ds = eval_val_ds.map(lambda x, y: (img_scaling(tf.cast(x, dtype=tf.float32)), y), tf.data.AUTOTUNE)
eval_val_ds = eval_val_ds.batch(BATCH_SIZE)
eval_val_ds = eval_val_ds.prefetch(tf.data.AUTOTUNE)

#eval_test_ds = tf.data.Dataset.from_tensor_slices((x_test , tf.keras.utils.to_categorical(y_test, NUMERO_CLASSES)))
eval_test_ds = tf.data.Dataset.from_tensor_slices((x_index , tf.keras.utils.to_categorical(y_index, NUMERO_CLASSES)))
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


# Pre-trained Uses the frozen pre-trained backbone from the ContrastiveModel and only trains the weights in the linear layer.
#pt_eval_model = get_eval_model(
#    _IMG_SIZE_X=IMG_SIZE_X,
#    _IMG_SIZE_Y = IMG_SIZE_Y,
#    _IMG_SIZE_D = IMG_SIZE_D,
#    backbone=contrastive_model.backbone,
#    total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
#    trainable=False,
#    lr=1e-3,
#)

#pt_eval_model.summary()
#pt_history = pt_eval_model.fit(
#    eval_train_ds,
#    batch_size=BATCH_SIZE,
#    epochs=TEST_EPOCHS,
#    steps_per_epoch=TEST_STEPS_PER_EPOCH,
#    validation_data=eval_val_ds,
#    validation_steps=VAL_STEPS_PER_EPOCH,
#)

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(no_pt_history.history["val_loss"])
plt.grid()
plt.title(f"loss")
        
plt.subplot(1, 2, 2)
plt.plot(no_pt_history.history["val_acc"], label="acc")
plt.grid()
plt.title(f"accuracy")
plt.legend()

plt.show()
plt.savefig(image_histortResult_FineTunnning)


# Comparison
no_pretrain = no_pt_eval_model.evaluate(eval_test_ds)
print("Without SSL ", no_pretrain)
log.info("Without SSL   loss: " + str(no_pretrain[0]) + "  Acc: " + str(no_pretrain[1]))


#pretrained = pt_eval_model.evaluate(eval_test_ds)
#print("With SSL ", pretrained)
#log.info("With SSL   loss: " + str(pretrained[0]) + "  Acc: " + str(pretrained[1]))
