import os
import dcase_util
import h5py
import matplotlib
import tensorflow as tf
import sys
sys.path.append('/home/dcase_2023/myDCASE/')  
from myLiB.utils import *

matplotlib.use('Agg')
# matplotlib.use('TKAgg')
import keras_tuner as kt


# Create timer instance
timer = dcase_util.utils.Timer()

scene_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

fs = 44100
hdf5_path = "waveTest_44100.h5"

with h5py.File(hdf5_path, 'r') as hf:
    print(hf.keys())
    X_train = np.array(hf['X_train'])
    X_validation = np.array(hf['X_validation'])
    Y_validation = [x.decode() for x in hf['Y_validation']]
    Y_train = [x.decode() for x in hf['Y_train']]

Y_train = labelsEncoding('Val', scene_labels, Y_train)
Y_validation = labelsEncoding('Val', scene_labels, Y_validation)

sr = fs
len_second = 1.0
new_X_train=[]
for i in range (len(X_train)):
    src = X_train[i][:int(sr * len_second)]
    src = np.expand_dims(src, axis=1)
    new_X_train.append(src)

input_shape = new_X_train[0].shape
print('The shape of an item', input_shape)

new_X_validation=[]
for i in range (len(X_validation)):
    src = X_validation[i][:int(sr * len_second)]
    src = np.expand_dims(src, axis=1)
    new_X_validation.append(src)

new_X_train = tf.stack(new_X_train)
Y_train = tf.stack(Y_train)
new_X_validation = tf.stack(new_X_validation)
Y_validation = tf.stack(Y_validation)

del X_train
del X_validation

# from kapre import STFT, Magnitude, MagnitudeToDecibel
def modelo(hp):
    tf.keras.backend.clear_session()
    from kapre.composed import get_melspectrogram_layer
    n_ffts = hp.Choice("n_ffts",  [256, 512, 1024, 2048],default=256)
    #n_mels = hp.Choice("n_mels", [20, 40, 60],default=40)
    n_mels = hp.Choice("n_mels",[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],default=20) #120,160,200

    # if(hp.get('n_ffts')==256):
    with hp.conditional_scope('n_ffts', [256]):
        hop1 = hp.Choice("hop_256", values=[(int(256 * 0.25)),(int(256 * 0.5)),(int(256 * 0.75)) ])
    # if (hp.get('n_ffts') == 512):
    with hp.conditional_scope('n_ffts', [512,256]):
        hop2 = hp.Choice("hop_512", values=[(int(512 * 0.25)), (int(512 * 0.5)),
                                           (int(512 * 0.75))])
    # if (hp.get('n_ffts') == 1024):
    with hp.conditional_scope('n_ffts', [1024,256]):
        hop3 = hp.Choice("hop_1024", values=[(int(1024 * 0.25)), (int(1024 * 0.5)),
                                           (int(1024 * 0.75))])
    # if (hp.get('n_ffts') == 2048):
    with hp.conditional_scope('n_ffts', [2048,256]):
        hop4 = hp.Choice("hop_2048", values=[(int(2048 * 0.25)), (int(2048 * 0.5)),
                                           (int(2048 * 0.75))])

    if(hp.get('n_ffts')==256):
        hop=hop1
    if(hp.get('n_ffts')==512):
        hop=hop2
    if(hp.get('n_ffts')==1024):
        hop=hop3
    if(hp.get('n_ffts')==2048):
        hop=hop4
    melgram = get_melspectrogram_layer(input_shape=input_shape,
                                       win_length=n_ffts,
                                       # win_length=2050,
                                       # hop_length=hop,
                                       hop_length=hop,
                                       sample_rate=fs,
                                       n_fft=n_ffts,
                                       # n_fft=2048,
                                       # return_decibel=hp.Choice("return_decibel", ["True", "False"]),
                                       return_decibel=True,
                                       n_mels=n_mels,
                                       # n_mels=40,
                                       mel_f_min=0.0,
                                       mel_f_max=fs / 2,
                                       input_data_format='channels_last',
                                       output_data_format='channels_last')

    #  ------ MODELO BASE ------ 
    model = tf.keras.Sequential()
    model.add(melgram)
    # model.add(spec_augment)
    # CNN layer #1:
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', name="Conv-1"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    #CNN layer #2:
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same',name="Conv-2"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    # CNN layer #3:
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', name="Conv-3"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(model.layers[11].output_shape[1] // model.layers[11].output_shape[1],model.layers[11].output_shape[2] // model.layers[11].output_shape[2])))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', name="lastLayer"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])  # ver dif entre categorical_accuracy vs accuracy

    return model

class MyTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        # kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 1, 65, step=4)
        # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 200)
        modelz = self.hypermodel.build(trial.hyperparameters)
        # modelz.summary()

        return super(MyTuner, self).run_trial(trial, *args, **kwargs)
try:
    with tf.device('gpu:0'):
        # tuner = MyTuner(
        #     hypermodel=modelo,
        #     directory='kerasTuner/RandomSearch',
        #     project_name='fs_' + str(fs),
        #     overwrite=False,
        #     oracle=kt.oracles.RandomSearch(
        #         objective=kt.Objective('val_loss', "min"),
        #         max_trials=10,
        #
        #     ))
        tuner = MyTuner(
        hypermodel=modelo,
        directory='kerasTuner/Hyperband',
        project_name='fs_'+str(fs),
        overwrite=False,
        oracle=kt.oracles.Hyperband(
            objective='val_loss',
            max_epochs=80,
            factor=3))

        #"kerasTuner/RandomSearch/" + 'fs_' + str(fs)
        directory="kerasTuner/Hyperband/"+"fs_"+str(fs)
        # Setup logging
        dcase_util.utils.setup_logging(logging_file=os.path.join(directory+"/task1a_v2_em.log"))
        # Get logging interface
        log = dcase_util.ui.ui.FancyLogger()
        log.title('DCASE2022 / Task1A -- Keras kapre wave tuning')
        log.line()

        # tracking_address = "kerasTuner/RandomSearch/" + 'fs_' + str(fs) + "/tmp/tb_logs"  # the path of your log file.
        tracking_address = directory+"/tmp/tb_logs" # the path of your log file.
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search_space_summary()

        tuner.search(new_X_train, Y_train,epochs=80, batch_size=64, shuffle=True,validation_data=(new_X_validation, Y_validation),callbacks=[stop_early,tf.keras.callbacks.TensorBoard(tracking_address)])
except Exception as e:
    log.line(e)
# Get the optimal hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# Get the optimal hyperparameters from the results
best_hps = tuner.get_best_hyperparameters()[0]
# Build model
log.line(str(best_hps.values))
h_model = tuner.hypermodel.build(best_hps)
h_model.summary(print_fn=log.info)
log.line("Get the optimal hyperparameters from the results")
print("")
