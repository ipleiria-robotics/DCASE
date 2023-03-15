import os

import h5py

from myLiB.funcoesPipline import *
from myLiB.models import *
from myLiB.plots import *
from myLiB.utils import *

matplotlib.use('Agg')
# matplotlib.use('TKAgg')
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

dasetName="kerasTuner8k_140_8_WavSpecAug"
hdf5_path_Train = "data/8k_140_8/WavSpecAug/Train_fs8000_140_2048_0.256_0.128.h5"
hdf5_path_Test = "data/8k_140_8/WavSpecAug/Test_fs8000_140_2048_0.256_0.128.h5"

directory="resultados/AI4EDGE_2/"
if os.path.isdir(directory): pass
else: os.makedirs(directory)
dcase_util.utils.setup_logging(logging_file=os.path.join(directory+"Emsemble_task1a_1TreinoDefault_2AllData.log"))
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2022 / Task1A -- low-complexity Acoustic Scene Classification')
log.line()



with h5py.File(hdf5_path_Train, 'r') as hf:
    print(hf.keys())
    X_train = np.array(hf['X_train'])
    X_validation = np.array(hf['X_validation'])
    Y_validation = [x.decode() for x in hf['Y_validation']]
    Y_train = [x.decode() for x in hf['Y_train']]
    descricao = [x.decode() for x in hf['descricao']]
    descricao = "".join([str(elem) for elem in descricao])
# with h5py.File(hdf5_path_Test, 'r') as hf:
#     print(hf.keys())
#     X_Test = np.array(hf['features'])
#     Y_Test = [x.decode() for x in hf['scene_label']]
# # Concatenate dados treino e validação com dados test
# X_train = np.concatenate((X_train, X_Test))
# Y_train = np.concatenate((Y_train, Y_Test))
# X_validation = np.concatenate((X_validation, X_Test))
# Y_validation = np.concatenate((Y_validation, Y_Test))
# del X_Test, Y_Test
def divide2Emsemle(x,y,ordenar ="ordenar1"):
    x1=[]
    y1 = []
    x2=[]
    y2 = []

    for i in range(len(y)):
        if(ordenar=="ordenar1"): #divisão em 5 grupos
            if (('airport' == y[i]) or('bus' == y[i]) or ('metro' == y[i] ) or ('metro_station' == y[i]) or ('street_pedestrian' == y[i])):
                x1.append(x[i])
                y1.append(y[i])
            else:
                x2.append(x[i])
                y2.append(y[i])
        if (ordenar == "ordenar2"): #dividir em 6 grupos
            if (('airport' == y[i]) or('bus' == y[i]) or ('metro' == y[i] ) or ('metro_station' == y[i]) or ('street_pedestrian' == y[i])):
                x1.append(x[i])
                y1.append(y[i])
            else:
                x1.append(x[i])
                y1.append("others")
            if (('park' == y[i]) or('public_square' == y[i]) or ('shopping_mall' == y[i] ) or ('street_traffic' == y[i]) or ('tram' == y[i])):
                x2.append(x[i])
                y2.append(y[i])
            else:
                x2.append(x[i])
                y2.append("others")
    return np.array(x1),np.array(y1),np.array(x2),np.array(y2)
x1,y1,x2,y2=divide2Emsemle(X_train, Y_train,"ordenar2")
# data = np.array([y1]).T
# df = pd.DataFrame(data, columns=['scene_label'])
# print(df['scene_label'].value_counts().to_string())
# data = np.array([y2]).T
# df = pd.DataFrame(data, columns=['scene_label'])
# print(df['scene_label'].value_counts().to_string())

scene_labels1 = ['airport',
                'bus',
                'metro',
                'metro_station',
                'street_pedestrian',
                 'others']
y1 = labelsEncoding('Train', scene_labels1, y1)
scene_labels2 = ['park',
                'public_square',
                'shopping_mall',
                'street_traffic',
                'tram',
                 'others']
y2 = labelsEncoding('Train', scene_labels2, y2)

x1 = np.expand_dims(x1, -1)
y1 = np.expand_dims(y1, -1)
x2 = np.expand_dims(x2, -1)
y2 = np.expand_dims(y2, -1)

x1Val,y1Val,x2Val,y2Val=divide2Emsemle(X_validation, Y_validation,"ordenar2")

y1Val = labelsEncoding('Val', scene_labels1, y1Val)
y2Val = labelsEncoding('Val', scene_labels2, y2Val)

x1Val = np.expand_dims(x1Val, -1)
y1Val = np.expand_dims(y1Val, -1)
x2Val = np.expand_dims(x2Val, -1)
y2Val = np.expand_dims(y2Val, -1)

del X_train,X_validation,Y_train,Y_validation

epocas = 200
batch_size = 64


try:
  keras_model1=loadModelH5(directory+"keras_model1.h5")
except:
    print("ainda não existe keras_model1")
    input_Shape = (x1.shape[1], x1.shape[2], 1)
    keras_model1=miniBM2_1(input_Shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model1.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    keras_model1.summary(print_fn=log.info)
    callback_list = [
        dcase_util.tfkeras.ProgressLoggerCallback(
            epochs=epocas,
            metric='categorical_accuracy',
            loss='categorical_crossentropy',
            output_type='logging'
        ),
        dcase_util.tfkeras.StasherCallback(
            epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    ]
    history=keras_model1.fit(x1, y1, epochs=epocas, batch_size=batch_size,validation_data=(x1Val, y1Val), shuffle=True, verbose=0,callbacks=callback_list)
    for callback in callback_list:
        if isinstance(callback, dcase_util.tfkeras.StasherCallback):
            # Fetch the best performing model
            callback.log()
            best_weights = callback.get_best()['weights']
            if best_weights:
                keras_model1.set_weights(best_weights)
            break
    keras_model1.save(directory+"keras_model1.h5")
    plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                 history.history['loss'],
                 history.history['val_loss'],
                 png_name=directory+"keras_model1.png")
    del x1,y1,x1Val,y1Val

try:
  keras_model2=loadModelH5(directory+"keras_model2.h5")
except:
    print("ainda não existe keras_model1")
    input_Shape = (x2.shape[1], x2.shape[2], 1)
    keras_model2=miniBM2_1(input_Shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    keras_model2.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])  # loss=tfr.keras.losses.SoftmaxLoss()
    keras_model2.summary(print_fn=log.info)
    callback_list = [
        dcase_util.tfkeras.ProgressLoggerCallback(
            epochs=epocas,
            metric='categorical_accuracy',
            loss='categorical_crossentropy',
            output_type='logging'
        ),
        dcase_util.tfkeras.StasherCallback(
            epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    ]
    history=keras_model2.fit(x2, y2, epochs=epocas, batch_size=batch_size,validation_data=(x2Val, y2Val), shuffle=True, verbose=0,callbacks=callback_list)
    for callback in callback_list:
        if isinstance(callback, dcase_util.tfkeras.StasherCallback):
            # Fetch the best performing model
            callback.log()
            best_weights = callback.get_best()['weights']
            if best_weights:
                keras_model2.set_weights(best_weights)
            break
    keras_model2.save(directory+"keras_model2.h5")
    plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                 history.history['loss'],
                 history.history['val_loss'],
                 png_name=directory+"keras_model2.png")
    del x2,y2,x2Val,y2Val



#Juntar e treinar o modelo final
with h5py.File(hdf5_path_Train, 'r') as hf:
    print(hf.keys())
    X_train = np.array(hf['X_train'])
    X_validation = np.array(hf['X_validation'])
    Y_validation = [x.decode() for x in hf['Y_validation']]
    Y_train = [x.decode() for x in hf['Y_train']]
    descricao = [x.decode() for x in hf['descricao']]
    descricao = "".join([str(elem) for elem in descricao])
with h5py.File(hdf5_path_Test, 'r') as hf:
    print(hf.keys())
    X_Test = np.array(hf['features'])
    Y_Test = [x.decode() for x in hf['scene_label']]
# Concatenate dados treino e validação com dados test
X_train = np.concatenate((X_train, X_Test))
Y_train = np.concatenate((Y_train, Y_Test))
X_validation = np.concatenate((X_validation, X_Test))
Y_validation = np.concatenate((Y_validation, Y_Test))
del X_Test, Y_Test
Y_train = labelsEncoding('Train', scene_labels, Y_train)
Y_validation = labelsEncoding('Val', scene_labels, Y_validation)

X_train = np.expand_dims(X_train, -1)
Y_train = np.expand_dims(Y_train, -1)
X_validation = np.expand_dims(X_validation, -1)
Y_validation = np.expand_dims(Y_validation, -1)

try:
  modelemsemble=loadModelH5(directory+"modelemsemble")
except:
    model1 = tf.keras.Sequential(name="model1")
    for layer in keras_model1.layers[:-1]:  # go through until last layer
        model1.add(layer)

    model2 = tf.keras.Sequential(name="model2")
    for layer in keras_model2.layers[:-1]:  # go through until last layer
        model2.add(layer)
    del keras_model2, keras_model1
    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    input = tf.keras.layers.Input(shape=input_Shape)
    x1=model1(input,training=False)
    x2=model2(input,training=False)
    concat = tf.keras.layers.Concatenate()([x1, x2])
    output = tf.keras.layers.Dense(10, activation='softmax')(concat)
    modelemsemble = tf.keras.Model(input, output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    modelemsemble.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    modelemsemble.summary(print_fn=log.info)
    callback_list = [
        dcase_util.tfkeras.ProgressLoggerCallback(
            epochs=epocas,
            metric='categorical_accuracy',
            loss='categorical_crossentropy',
            output_type='logging'
        ),
        dcase_util.tfkeras.StasherCallback(
            epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    ]
    history=modelemsemble.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=0,callbacks=callback_list)
    for callback in callback_list:
        if isinstance(callback, dcase_util.tfkeras.StasherCallback):
            # Fetch the best performing model
            callback.log()
            best_weights = callback.get_best()['weights']
            if best_weights:
                modelemsemble.set_weights(best_weights)
            break
    modelemsemble.save(directory+"modelemsemble")


    plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                 history.history['loss'],
                 history.history['val_loss'],
                 png_name=directory+"modelemsemble.png")

    # test
    # seleciona 50% dos dados para quantizar em INT8
    random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
    representative_data = X_train[random_indices, :]

    x_test_normalized = representative_data
    def representative_dataset():
        for x in x_test_normalized:
            yield [np.array([x], dtype=np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(modelemsemble)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
    converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    with open(directory+"modelemsemble.tflite", "wb") as output_file:
        output_file.write(tflite_model)


# macc, params = nessi.get_model_size(directory+"modelemsemble.tflite", 'tflite')
# nessi.validate(macc, params, log)

log.section_header('testing')
timer.start()
testNameDir=directory
fold_model_filename="modelemsemble.tflite"
path_estimated_scene = "res_fold_0.csv"
if not os.path.isfile(testNameDir+"/"+path_estimated_scene):
    # Loop over all cross-validation folds and learn acoustic models
    with h5py.File(hdf5_path_Test, 'r') as hf:
        print(hf.keys())
        # features = int16_to_float32(hf['features'][index_files])
        test_features = np.array(hf['features'])
        test_filename = [x.decode() for x in hf['filename']]
        # scene_label = [x.decode() for x in hf['scene_label']]

    do_testing(scene_labels, fold_model_filename, path_estimated_scene, test_features, test_filename, log,testNameDir)
timer.stop()
log.foot(time=timer.elapsed())

log.section_header('evaluation')
timer.start()
# do_evaluation(path_estimated_scene,log)
df = do_evaluation(path_estimated_scene,log,testNameDir,fold_model_filename)
timer.stop()
log.foot(time=timer.elapsed())
