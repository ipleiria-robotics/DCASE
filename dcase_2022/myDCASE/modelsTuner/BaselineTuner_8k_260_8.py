import os

import dcase_util
import h5py
import matplotlib
import tensorflow as tf

# from myLiB.funcoesPipline import *
from myLiB.utils import *

# from myLiB.models import *
# from myLiB.plots import *
matplotlib.use('Agg')
# matplotlib.use('TKAgg')
import keras_tuner as kt


# Create timer instance
timer = dcase_util.utils.Timer()

scene_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

# kerasTuner8k_run1
dasetName="kerasTuner8k_260_8"
hdf5_path_Train = "data/8k_260_8/Train_fs8000_260_2048_0.256_0.128.h5"
hdf5_path_Test = "data/8k_260_8/Test_fs8000_260_2048_0.256_0.128.h5"


with h5py.File(hdf5_path_Train, 'r') as hf:
    print(hf.keys())
    # X_train = np.array(hf['X_train'])
    # Y_train = [x.decode() for x in hf['Y_train']]
    X_train = np.array(hf['X_validation'])
    Y_train = [x.decode() for x in hf['Y_validation']]

with h5py.File(hdf5_path_Test, 'r') as hf:
    print(hf.keys())
    X_validation = np.array(hf['features'])
    Y_validation = [x.decode() for x in hf['scene_label']]


Y_train = labelsEncoding('Train', scene_labels,Y_train)  # 'train'= 'smooth_labels'|'Mysmooth_labels'|'Mysmooth_labels2'
Y_validation = labelsEncoding('Val', scene_labels,Y_validation)  # 'train'= 'smooth_labels'|'Mysmooth_labels'|'Mysmooth_labels2'

epocas=200
batch_size = 64
datasetTrain = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(Y_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
datasetValid = tf.data.Dataset.from_tensor_slices((X_validation, Y_validation)).shuffle(len(Y_validation)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# datasetTrain = tf.data.Dataset.zip(tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(Y_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))
# datasetValid = tf.data.Dataset.zip(tf.data.Dataset.from_tensor_slices((X_validation, Y_validation)).shuffle(len(Y_validation)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))
del X_train, Y_train , X_validation, Y_validation,

def modelo(hp):
    tf.keras.backend.clear_session()
    filterL1 = hp.Int("filterL1", min_value=4, max_value=32, step=4)
    filterL2 = hp.Int("filterL2", min_value=4, max_value=48, step=4)
    filterL3 = hp.Int("filterL3", min_value=4, max_value=64, step=4)
    dropout1 = hp.Float('dropout1', 0, 0.7, step=0.1)
    dropout2 = hp.Float('dropout2', 0, 0.7, step=0.1)
    dropout3 = hp.Float('dropout3', 0, 0.7, step=0.1)
    CNN_kernel_size_1 = hp.Choice("CNN_kernel_size_1", [3, 5, 7])
    CNN_kernel_size_11 = hp.Choice("CNN_kernel_size_11",[3, 5, 7])
    CNN_kernel_size_2 = hp.Choice("CNN_kernel_size_2", [3, 5, 7])
    CNN_kernel_size_22 = hp.Choice("CNN_kernel_size_22",[3, 5, 7])
    CNN_kernel_size_3 = hp.Choice("CNN_kernel_size_3", [3, 5, 7])
    CNN_kernel_size_33 = hp.Choice("CNN_kernel_size_33",[3, 5, 7])
    pool_kernel_size_2 = hp.Choice("pool_kernel_size_2", [1,2])
    pool_kernel_size_22 = hp.Choice("pool_kernel_size_22",[1,2])
    pool_kernel_size_3 = hp.Choice("pool_kernel_size_3", [1,2])
    pool_kernel_size_33 = hp.Choice("pool_kernel_size_33",[1,2])
    pooling_1 = hp.Choice('pooling_1', ['avg', 'max'])
    pooling_2 = hp.Choice('pooling_2', ['avg', 'max'])
    pooling_3 = hp.Choice('pooling_3', ['avg', 'max',"clear"])
    units = hp.Int("units", min_value=32, max_value=256, step=32)

    # input_Shape=(X_train.shape[1],X_train.shape[2],1) [freq,time]
    input_Shape = (datasetTrain.element_spec[0].shape[1], datasetTrain.element_spec[0].shape[2], 1)
    input = tf.keras.layers.Input(shape=input_Shape)
    # CNN layer #1:
    x = tf.keras.layers.Conv2D(filters=filterL1, kernel_size=(CNN_kernel_size_1, CNN_kernel_size_11), padding='same', name="Conv-1")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #CNN layer #2:
    x = tf.keras.layers.Conv2D(filters=filterL2, kernel_size=(CNN_kernel_size_2, CNN_kernel_size_22), activation='relu', padding='same',name="Conv-2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_1 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_2, pool_kernel_size_22))(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    # CNN layer #3:
    x = tf.keras.layers.Conv2D(filters=filterL3, kernel_size=(CNN_kernel_size_3, CNN_kernel_size_33), activation='relu', padding='same', name="Conv-3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if pooling_2 == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    else:
        x = tf.keras.layers.AveragePooling2D(pool_size=(pool_kernel_size_3, pool_kernel_size_33))(x)
    x = tf.keras.layers.Dropout(dropout2)(x)

    for i in range(hp.Int("num_layers", 0, 1)):
        x = tf.keras.layers.Conv2D(filters=hp.Int(f'newfilter_{i}', min_value=4, max_value=32, step=4), kernel_size=(hp.Choice(f'kernel1_{i}', [3, 5, 7]), hp.Choice(f'kernel2_{i}', [3, 5, 7])),
                                   activation='relu', padding='same', name=f"Conv-3_{i}")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        if hp.Choice(f'newPooling_{i}', ['avg', 'max']) == 'max':
            x = tf.keras.layers.MaxPooling2D(pool_size=(hp.Choice(f'pool1_{i}',[1,2]), hp.Choice(f'pool2_{i}',[1,2])))(x)
        else:
            x = tf.keras.layers.AveragePooling2D(pool_size=(hp.Choice(f'pool1_{i}',[1,2]), hp.Choice(f'pool2_{i}',[1,2])))(x)
        x = tf.keras.layers.Dropout(hp.Float(f'newdropout_{i}', 0, 0.7, step=0.1))(x)

    if pooling_3 == 'max':
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif pooling_3 == 'avg':
        x = tf.keras.layers.GlobalAvgPool2D()(x)
    elif pooling_3 == 'clear':
        pass

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu', name="lastLayer")(x)
    x = tf.keras.layers.Dropout(dropout3)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(input, output)

    # hp_optimizer = hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop'])
    # optimizer = tf.keras.optimizers.get(hp_optimizer)
    optimizer = tf.keras.optimizers.get('adam')
    # optimizer.learning_rate = hp.Choice("learning_rate", [0.1, 0.01, 0.001,0.0001], default=0.01)
    optimizer.learning_rate = 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

class MyTuner(kt.engine.tuner.Tuner):
    def maybe_compute_model_size(self, model):
        """Compute the size of a given model, if it has been built."""
        if model.built:
            params = [tf.keras.backend.count_params(p) for p in model.trainable_weights]
            return int(np.sum(params))
        return 0
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        # kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 1, 65, step=4)
        # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 200)
        #
        modelz = self.hypermodel.build(trial.hyperparameters)
        MAX_MACC = 30e6  # 30M MACC
        MAX_PARAMS = 128e3  # 128K params
        model_size = self.maybe_compute_model_size(modelz)
        print("Considering model with size: {}".format(model_size))
        #calculate macc
        config = modelz.get_config()  # Returns pretty much every information about your model
        in_sz = np.array(config["layers"][0]["config"]["batch_input_shape"])
        in_sz[in_sz == None] = 1
        print(in_sz)
        modelz.save('tmp.h5')
        tf.compat.v1.reset_default_graph()
        def get_maccs(model_h5_path, in_size):
            session = tf.compat.v1.Session()
            graph = tf.compat.v1.get_default_graph()
            with graph.as_default():
                with session.as_default():
                    model = tf.keras.models.load_model(model_h5_path)(np.ones(in_size))
                    run_meta = tf.compat.v1.RunMetadata()
                    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                    flops = tf.compat.v1.profiler.profile(graph=graph,
                                                          run_meta=run_meta, cmd='op', options=opts)
                    return flops.total_float_ops // 2
        macc = get_maccs('tmp.h5', in_sz)
        print(macc)

        if model_size > MAX_PARAMS or macc > MAX_MACC :
            # self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.INVALID)
            self.oracle.end_trial(trial.trial_id, status="INVALID")
            # dummy_history_obj = tf.keras.callbacks.History()
            # dummy_history_obj.on_train_begin()
            # dummy_history_obj.history.setdefault('val_loss', []).append(2.5)
            return  0
        # modelz.summary()
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)
    def on_trial_end(self, trial):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        if not trial.get_state().get("status") == "INVALID":
            self.oracle.end_trial(trial.trial_id, status="COMPLETED")
            # self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.COMPLETED)

        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()


tuner = MyTuner(
hypermodel=modelo,
directory='kerasTuner/Hyperband/Baseline',
project_name='8k_260_8_test1',
overwrite=False,
oracle=kt.oracles.Hyperband(
    objective='val_loss',
    max_epochs=epocas,
    factor=3))

directory = "kerasTuner/Hyperband/Baseline/"+'8k_260_8_test1'
# Setup logging
dcase_util.utils.setup_logging(logging_file=os.path.join(directory + "/task1a_v2_em.log"))
# Get logging interface
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2022 / Task1A -- Keras kapre wave tuning')
log.line()

tracking_address = directory+"/tmp/tb_logs" # the path of your log file.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tuner.search_space_summary()

tuner.search(datasetTrain, epochs=epocas,validation_data=datasetValid, batch_size=batch_size,callbacks=[stop_early, tf.keras.callbacks.TensorBoard(tracking_address)])
# tuner.search(X_train, Y_train, epochs=250, batch_size=64, shuffle=True,
#              validation_data=(X_validation, Y_validation),
#              callbacks=[stop_early, tf.keras.callbacks.TensorBoard(tracking_address)])

# Get the optimal hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# Get the optimal hyperparameters from the results
best_hps = tuner.get_best_hyperparameters()[0]
log.line(str(best_hps.values))
# Build model
h_model = tuner.hypermodel.build(best_hps)
h_model.summary(print_fn=log.info)
log.line("Get the optimal hyperparameters from the results")
print("")
