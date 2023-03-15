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


directory="resultados/AI4EDGE_3/"
if os.path.isdir(directory): pass
else: os.makedirs(directory)
dcase_util.utils.setup_logging(logging_file=os.path.join(directory+"EmsembleKD_task1a_v2.log"))
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2022 / Task1A -- low-complexity Acoustic Scene Classification')
log.line()

#Fazer um emsemble de 10 modelos de modo a criar um teacher e por KD criar um modelo de menor dimensão
def divide2Emsemle(x,y,label):
    x1=[]
    y1 = []
    for i in range(len(y)):
            if ((label == y[i])):
                x1.append(x[i])
                y1.append(y[i])
            else:
                x1.append(x[i])
                y1.append("others")
    return np.array(x1),np.array(y1)

def readData(hdf5_path_Train,hdf5_path_Test):
    with h5py.File(hdf5_path_Train, 'r') as hf:
        print(hf.keys())
        X_train = np.array(hf['X_train'])
        X_validation = np.array(hf['X_validation'])
        Y_validation = [x.decode() for x in hf['Y_validation']]
        Y_train = [x.decode() for x in hf['Y_train']]
        descricao = [x.decode() for x in hf['descricao']]
        descricao = "".join([str(elem) for elem in descricao])

    return X_train,X_validation,Y_validation,Y_train

epocas = 200
batch_size = 64

for x in range(len(scene_labels)):
    modelName=f"keras_model_{x}.h5"
    label = scene_labels[x]
    try:
      globals()[f"keras_model_{x}"]=loadModelH5(directory+modelName)
    except:
        print("ainda não existe "+ modelName)
        X_train, X_validation, Y_validation, Y_train = readData(hdf5_path_Train, hdf5_path_Test)
        x1, y1 = divide2Emsemle(X_train, Y_train, label)
        # data = np.array([y1]).T
        # df = pd.DataFrame(data, columns=['scene_label'])
        # print(df['scene_label'].value_counts().to_string())
        scene_labels1 = [label, 'others']
        y1 = labelsEncoding('Train', scene_labels1, y1)
        x1 = np.expand_dims(x1, -1)
        y1 = np.expand_dims(y1, -1)
        x1Val, y1Val = divide2Emsemle(X_validation, Y_validation, label)
        y1Val = labelsEncoding('Val', scene_labels1, y1Val)
        x1Val = np.expand_dims(x1Val, -1)
        y1Val = np.expand_dims(y1Val, -1)
        del X_train, X_validation, Y_train, Y_validation

        input_Shape = (x1.shape[1], x1.shape[2], 1)
        keras_model=BM2_emKD(input_Shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
        keras_model.summary(print_fn=log.info)
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
        history=keras_model.fit(x1, y1, epochs=epocas, batch_size=batch_size,validation_data=(x1Val, y1Val), shuffle=True, verbose=0,callbacks=callback_list)
        for callback in callback_list:
            if isinstance(callback, dcase_util.tfkeras.StasherCallback):
                # Fetch the best performing model
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    keras_model.set_weights(best_weights)
                break
        keras_model.save(directory+modelName)
        plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                     history.history['loss'],
                     history.history['val_loss'],
                     png_name=directory+"keras_model"+str(x)+".png")
        del x1,y1,x1Val,y1Val

print("")

with h5py.File(hdf5_path_Train, 'r') as hf:
    print(hf.keys())
    X_train = np.array(hf['X_train'])
    X_validation = np.array(hf['X_validation'])
    Y_validation = [x.decode() for x in hf['Y_validation']]
    Y_train = [x.decode() for x in hf['Y_train']]
    descricao = [x.decode() for x in hf['descricao']]
    descricao = "".join([str(elem) for elem in descricao])
Y_train = labelsEncoding('Train', scene_labels, Y_train)
Y_validation = labelsEncoding('Val', scene_labels, Y_validation)

X_train = np.expand_dims(X_train, -1)
Y_train = np.expand_dims(Y_train, -1)
X_validation = np.expand_dims(X_validation, -1)
Y_validation = np.expand_dims(Y_validation, -1)

try:
    for x in range(len(scene_labels)):
        try:
            globals()[f"keras_model_{x}"] = loadModelH5(directory + modelName)
        except:
            print("An exception occurred")
    teacher=loadModelH5(directory+"teacher")
except:
    model0 = tf.keras.Sequential(name="model0")
    for layer in keras_model_0.layers[:-1]:  # go through until last layer
        model0.add(layer)
    model1 = tf.keras.Sequential(name="model1")
    for layer in keras_model_1.layers[:-1]:
        model1.add(layer)
    model2 = tf.keras.Sequential(name="model2")
    for layer in keras_model_2.layers[:-1]:
        model2.add(layer)
    model3 = tf.keras.Sequential(name="model3")
    for layer in keras_model_3.layers[:-1]:  #
        model3.add(layer)
    model4 = tf.keras.Sequential(name="model4")
    for layer in keras_model_4.layers[:-1]:
        model4.add(layer)
    model5 = tf.keras.Sequential(name="model5")
    for layer in keras_model_5.layers[:-1]:
        model5.add(layer)
    model6 = tf.keras.Sequential(name="model6")
    for layer in keras_model_6.layers[:-1]:
        model6.add(layer)
    model7 = tf.keras.Sequential(name="model7")
    for layer in keras_model_7.layers[:-1]:
        model7.add(layer)
    model8 = tf.keras.Sequential(name="model8")
    for layer in keras_model_8.layers[:-1]:
        model8.add(layer)
    model9 = tf.keras.Sequential(name="model9")
    for layer in keras_model_9.layers[:-1]:
        model9.add(layer)
    del keras_model_0, keras_model_1, keras_model_2, keras_model_3, keras_model_4, keras_model_5, keras_model_6, keras_model_7, keras_model_8, keras_model_9

    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    input = tf.keras.layers.Input(shape=input_Shape)
    x0=model0(input,training=False)
    x1=model1(input,training=False)
    x2=model2(input,training=False)
    x3=model3(input,training=False)
    x4=model4(input,training=False)
    x5=model5(input,training=False)
    x6=model6(input,training=False)
    x7=model7(input,training=False)
    x8=model8(input,training=False)
    x9=model9(input,training=False)
    concat = tf.keras.layers.Concatenate()([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9])
    output = tf.keras.layers.Dense(10, activation='softmax')(concat)
    teacher = tf.keras.Model(input, output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    teacher.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    teacher.summary(print_fn=log.info)
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
    history=teacher.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=0,callbacks=callback_list)
    for callback in callback_list:
        if isinstance(callback, dcase_util.tfkeras.StasherCallback):
            # Fetch the best performing model
            callback.log()
            best_weights = callback.get_best()['weights']
            if best_weights:
                teacher.set_weights(best_weights)
            break
    teacher.save(directory+"teacher")

    plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                 history.history['loss'],
                 history.history['val_loss'],
                 png_name=directory+"teacher.png")

random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
representative_data = X_train[random_indices, :]

x_test_normalized = representative_data
def representative_dataset():
    for x in x_test_normalized:
        yield [np.array([x], dtype=np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(teacher)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open(directory+"teacher.tflite", "wb") as output_file:
    output_file.write(tflite_model)

log.section_header('testing-teacher')
timer.start()
testNameDir=directory
fold_model_filename="teacher.tflite"
path_estimated_scene = "res_fold_teacher.csv"
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

log.section_header('evaluation-teacher')
timer.start()
# do_evaluation(path_estimated_scene,log)
df = do_evaluation(path_estimated_scene,log,testNameDir,fold_model_filename)
timer.stop()
log.foot(time=timer.elapsed())


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

try:
  student=loadModelH5(directory+"student")
except:
    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    student = BM2(input_Shape)
    student._name = "Student"
    # Clone student for later comparison
    student_scratch = tf.keras.models.clone_model(student)

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
        metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    callback_list = [
        dcase_util.tfkeras.StasherCallback(
            epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_student_loss', patience=20, verbose=0, mode='min')
    ]
    # Distill teacher to student
    history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

    student = distiller.student
    student.save(directory+"student")

# test
# seleciona 50% dos dados para quantizar em INT8
random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
representative_data = X_train[random_indices, :]

x_test_normalized = representative_data
def representative_dataset():
    for x in x_test_normalized:
        yield [np.array([x], dtype=np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open(directory+"student.tflite", "wb") as output_file:
    output_file.write(tflite_model)

#não coloquei no pc da estg
# macc, params = nessi.get_model_size(directory+"student.tflite", 'tflite')
# nessi.validate(macc, params, log)

log.section_header('testing')
timer.start()
testNameDir=directory
fold_model_filename="student.tflite"
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
