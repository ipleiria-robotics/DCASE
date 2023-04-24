import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ['TF_ENABLE_ONEDNN_OPTS']="1"
import h5py
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
#
# if len(physical_devices) > 0:
#     # Configure the GPU devices
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
#         tf.config.experimental.set_virtual_device_configuration(device, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

from myLiB.funcoesPipline import *
from myLiB.models import *
from myLiB.plots import *
from myLiB.utils import *
from myLiB import tcl

print("Tensorflow version")
print(tf.__version__)
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
current_dir=os.getcwd()
current_dir=current_dir.replace("\\","/")
dasetName="kerasTuner8k_140_8_WavSpecAug"
#hdf5_path_Train = "data/8k_140_8/WavSpecAug/Train_fs8000_140_2048_0.256_0.128.h5"
#hdf5_path_Test = "data/8k_140_8/WavSpecAug/Test_fs8000_140_2048_0.256_0.128.h5"
hdf5_path_Train = current_dir +"/dcase_2023/myDCASE/data/8k_140_8/WavSpecAug/Train_fs8000_140_2048_0.256_0.128.h5"
hdf5_path_Test = current_dir +"/dcase_2023/myDCASE/data/8k_140_8/WavSpecAug/Test_fs8000_140_2048_0.256_0.128.h5"

#directory="resultados/AI4EDGE_4/"
directory=current_dir +"/dcase_2023/myDCASE/resultados/AI4EDGE_4/"

if os.path.isdir(directory): pass
else: os.makedirs(directory)
dcase_util.utils.setup_logging(logging_file=os.path.join(directory+"EmsembleKD_task1a_1TrainDefault_2TrainAllData.log"))
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2022 / Task1A -- low-complexity Acoustic Scene Classification')
log.line("1TrainDefault_2TrainAllData")
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

#epocas = 200
epocas = 2
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

print("Teacher loaded")
random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
representative_data = X_train[random_indices, :]

x_test_normalized = representative_data
def representative_dataset():
    for x in x_test_normalized:
        yield [np.array([x], dtype=np.float32)]


testNameDir=directory
fold_model_filename="teacher.tflite"
path_teacher = testNameDir + "/" + fold_model_filename
# if teacher model isnt quantized
if not os.path.isfile(path_teacher):
    print("Converter teacher para int8")

    converter = tf.lite.TFLiteConverter.from_keras_model(teacher)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
    converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    with open(directory+"teacher.tflite", "wb") as output_file:
        output_file.write(tflite_model)

print("Testar teacher")
log.section_header('testing-teacher')
timer.start()


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

print("Avaliar teacher")
print("----FIX avaliar teacher")
# log.section_header('evaluation-teacher')
# timer.start()
# #df = do_evaluation(path_estimated_scene,log)
# df = do_evaluation(path_estimated_scene,log,testNameDir,fold_model_filename)
# timer.stop()
# log.foot(time=timer.elapsed())


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
        print("Teacher predicted")
        print(teacher_predictions)
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

print("Knowledge distillation start")
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
        temperature=3,
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

print("Quantizar o dataset started")
# test
# seleciona 50% dos dados para quantizar em INT8
random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
representative_data = X_train[random_indices, :]

x_test_normalized = representative_data
def representative_dataset():
    for x in x_test_normalized:
        yield [np.array([x], dtype=np.float32)]

fold_model_filename="student.tflite"
path_student = testNameDir + "/" + fold_model_filename
# if teacher model isnt quantized
if not os.path.isfile(path_student):
    print("Convertendo student")
    converter = tf.lite.TFLiteConverter.from_keras_model(student)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
    converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    with open(directory+"student.tflite", "wb") as output_file:
        output_file.write(tflite_model)


# macc, params = nessi.get_model_size(directory+"student.tflite", 'tflite')
# nessi.validate(macc, params, log)
print("Testing student")
log.section_header('testing')
timer.start()
testNameDir=directory

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

##############################
#   EVALUATION COMMENTED

# log.section_header('evaluation')
# timer.start()
# # do_evaluation(path_estimated_scene,log)
# df = do_evaluation(path_estimated_scene,log,testNameDir,fold_model_filename)
# timer.stop()
# log.foot(time=timer.elapsed())
##############################


#---------------------ADDED SECTION (CARLOS)-------------------------

#load student model

#load teacher model

#test teacher model

#test student model

print("Program done")


train_student_with_relational_kd=False
train_student_with_relational_response_kd=False
train_student_with_relational_resp_v2_kd=False
train_student_with_fitnet=True
optuna_trials = 50
train_optuna=False
shuffle_training_data=False
#---------------RELATION-BASED KD-----------------------

def Huber_loss(x, y):
    return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x - y), 1.), tf.square(x - y) / 2, tf.abs(x - y) - 1 / 2))

def Distance_wise_potential(x):
    x_square = tf.reduce_sum(tf.square(x), -1)
    prod = tf.matmul(x, x, transpose_b=True)
    distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square, 1) + tf.expand_dims(x_square, 0) - 2 * prod, 1e-12))
    mu = tf.reduce_sum(distance) / tf.reduce_sum(tf.cast(distance > 0., tf.float32))
    return distance / (mu + 1e-8)

def Distance_wise_potential_teste(x):
    x_square = tf.reduce_sum(tf.square(x), -1)
    prod = tf.matmul(x, x, transpose_b=True)
    # A^2 + B^2
    a2_b2 = tf.expand_dims(x_square, 2) + tf.expand_dims(x_square, 1)
    print(a2_b2)
    print(prod)
    # sqrt(a^2 + b^2 -2ab )
    distance = tf.sqrt(tf.maximum(a2_b2 - 2 * prod, 1e-12))
    mu = tf.reduce_sum(distance) / tf.reduce_sum(tf.cast(distance > 0., tf.float32))
    return distance / (mu + 1e-8)

def Angle_wise_potential(x):
    e = tf.expand_dims(x, 0) - tf.expand_dims(x, 1)
    e_norm = tf.nn.l2_normalize(e, 2)
    return tf.matmul(e_norm, e_norm, transpose_b=True)

class Distiller_relation(tf.keras.Model):
    def __init__(self, student, teacher,alpha_rel):
        super(Distiller_relation, self).__init__()
        self.teacher = teacher
        self.student = student
        self.alpha_rel = alpha_rel
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
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
        super(Distiller_relation, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        print("Teacher predicted")
        print(teacher_predictions)
        print("Alpha is "+str(self.alpha))
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            """ distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) """

            s = tf.nn.l2_normalize(student_predictions, 1)
            t = tf.nn.l2_normalize(teacher_predictions, 1)

            distance_loss = Huber_loss(Distance_wise_potential(s), Distance_wise_potential(t))
            angle_loss = Huber_loss(Angle_wise_potential(s), Angle_wise_potential(t))

            distillation_loss = (distance_loss + angle_loss) * 20
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

if train_student_with_relational_kd:
    print("TRAINING STUDENT WITH RELATIONAL KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "Student_rel_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)
        # Initialize and compile distiller
        distiller = Distiller_relation(student=student_rel_kd, teacher=teacher,alpha_rel=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_kd,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        # Distill teacher to student
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_rel_kd = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rel_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_w_rel_kd'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_w_rel_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_rel_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rel_kd")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in relation-based kd with AI4EDGE_4 v2",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "Student_rel_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = 0.5
        # Initialize and compile distiller
        distiller = Distiller_relation(student=student_rel_kd, teacher=teacher,alpha_rel=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_kd,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        # Distill teacher to student
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_rel_kd = distiller.student
        student_rel_kd.save(directory+"student_rel_kd")

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rel_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        if not os.path.isfile(path_student):
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_rel_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)

#---------------RESPONSE AND RELATION-BASED KD-----------------------

class Distiller_relation_response(tf.keras.Model):
    def __init__(self, student, teacher,alpha_rel):
        super(Distiller_relation_response, self).__init__()
        self.teacher = teacher
        self.student = student
        self.alpha_rel = alpha_rel
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
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
        super(Distiller_relation_response, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        print("Teacher predicted")
        print(teacher_predictions)
        print("Alpha is "+str(self.alpha))
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss_part1 = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

            s = tf.nn.l2_normalize(student_predictions, 1)
            t = tf.nn.l2_normalize(teacher_predictions, 1)

            distance_loss = Huber_loss(Distance_wise_potential(s), Distance_wise_potential(t))
            angle_loss = Huber_loss(Angle_wise_potential(s), Angle_wise_potential(t))

            distillation_loss_part2 = (distance_loss + angle_loss) * 20

            distillation_loss = distillation_loss_part1 + distillation_loss_part2

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

if train_student_with_relational_response_kd:
    print("TRAINING STUDENT WITH RELATIONAL + RESPONSE KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "student_rel_resp_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_resp_kd = trial.suggest_float('alpha_rel_resp_kd', 0.4, 0.6)
        # Initialize and compile distiller
        distiller = Distiller_relation_response(student=student_rel_kd, teacher=teacher,alpha_rel=alpha_rel_resp_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_resp_kd,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        # Distill teacher to student
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_rel_kd = distiller.student
        student_rel_kd.save(directory+student_rel_kd._name)

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename=student_rel_kd._name+".tflite"
        path_student = testNameDir + "/" + fold_model_filename

        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)
        acc_stud = metrics_stud[1]


        data = {
                'Network': ['student_w_rel_resp_kd'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_w_rel_resp_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+fold_model_filename, "wb") as output_file:
                output_file.write(tflite_model)
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in relation + response based kd with AI4EDGE_4 v2",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "student_rel_resp_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd=0.5
        # Initialize and compile distiller
        distiller = Distiller_relation_response(student=student_rel_kd, teacher=teacher,alpha_rel=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        # Distill teacher to student
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_rel_kd = distiller.student
        student_rel_kd.save(directory+student_rel_kd._name)

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename=student_rel_kd._name+".tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        if not os.path.isfile(path_student):
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+fold_model_filename, "wb") as output_file:
                output_file.write(tflite_model)

#---------------FITNET KD-----------------------
def define_intermediate_networks():
    teacher0=loadModelH5(directory+"keras_model_0.h5")
    #teacher_0_intermediate = tf.keras.Sequential(name="teacher0_intermediate")
    #for layer in teacher0.layers[:-1]:  # go through until last layer
    """ for layer in teacher0.layers[:4]:  # go through until 4th layer
        teacher_0_intermediate.add(layer)  """
    layer_name = 'Conv-2'
    teacher_0_intermediate = tf.keras.Model(inputs=teacher0.input,outputs=teacher0.get_layer(layer_name).output)
    teacher_0_intermediate.summary()


    

    teacher_0_intermediate._name = "teacher_0_intermediate"
    """ 
    for i, layer in enumerate(teacher0.layers):
        if i <= 3:  
            print("Layer of teacher is ",layer.name)
            print("Layer of teacher intermd is ",teacher_0_intermediate.layers[i-1].name)
            
            teacher_0_intermediate.layers[i-1].set_weights(layer.get_weights())

    teacher_0_intermediate.trainable = False
    teacher_0_intermediate.summary() """
    del teacher0
    print("Teachers loaded")

class Distiller_fitnet(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller_fitnet, self).__init__()

        self.teacher = teacher
        self.student = student

        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = True))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_intermediate_v4.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        
        self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_intermediate_v4.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])
 
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
        """ Configure the Distiller_fitnet.
    student_loss_fn: Loss function of difference between student
                predictions and ground-truth
    distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
    alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
    temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller_fitnet, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def train_step(self, data):

        global teacher_features, teacher_output
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        teacher_features = teacher_intermediate_v4(x, training=False)


        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate_v4(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            stud_feat_augmented = aux_pool(student_features, training=False)

            teach_feat_augmented = aux(teacher_features, training=False)

            diff = stud_feat_augmented - teach_feat_augmented
            
            mask = tf.cast((stud_feat_augmented > teach_feat_augmented) | (teach_feat_augmented > 0), tf.float32)
            
            # Convert the boolean mask to a float32 tensor before multiplying with the squared difference
            mask = tf.cast(mask, tf.float32)

            loss_value = tf.square(diff) * mask
            
            distillation_loss_later = tf.reduce_sum(loss_value)
            x_shape=tf.cast(tf.shape(x)[0], tf.float32)
            distillation_loss =  distillation_loss_later / x_shape * 1e-4

            print("Distillation loss is")
            print(distillation_loss)
            # alpha is the weight of the student loss vs the distillation loss (0 - 1)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            print("Alpha is", self.alpha)
            print("Loss in Distiller_fitnet :", loss)
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`
            self.compiled_metrics.update_state(y, student_prediction)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
            print("Train...", results)
            return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        ## Compute predictions
        y_prediction = self.student(x, training=False)

        # calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        print("Test...", results)
        return results

    def call(self, inputs):
        # Pass the input through the teacher and student models
        teacher_predictions = self.teacher(inputs)
        student_predictions = self.student(inputs)
        return teacher_predictions, student_predictions

class Distiller_response(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller_response, self).__init__()

        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
    
        super(Distiller_response, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha


    def train_step(self, data):
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_prediction)


            # Distillation loss for response-based knowledge
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_prediction / self.temperature, axis=1),
                tf.nn.softmax(student_prediction / self.temperature, axis=1)
            )
            """ distillation_loss = self.distillation_loss_fn(
                teacher_prediction / self.temperature,
                student_prediction / self.temperature
                ) """
            print("Distillation loss is")
            print(distillation_loss)
            # alpha is the weight of the student loss vs the distillation loss (0 - 1)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            print("Alpha is", self.alpha)
            print("Loss in distiller :", loss)
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`
            self.compiled_metrics.update_state(y, student_prediction)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
            print("Train...", results)
            return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        ## Compute predictions
        y_prediction = self.student(x, training=False)

        # calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        print("Y is:", y)
        print("Y_predicted is",y_prediction)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        print("Test...", results)
        return results

    def call(self, inputs):
        # Pass the input through the teacher and student models
        teacher_predictions = self.teacher(inputs)
        student_predictions = self.student(inputs)
        return teacher_predictions, student_predictions

class Distiller_response(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_response, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
    
        super(Distiller_response, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha


    def train_step(self, data):
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_prediction)


            # Distillation loss for response-based knowledge
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_prediction / self.temperature, axis=1),
                tf.nn.softmax(student_prediction / self.temperature, axis=1)
            )
            """ distillation_loss = self.distillation_loss_fn(
                teacher_prediction / self.temperature,
                student_prediction / self.temperature
                ) """
            print("Distillation loss is")
            print(distillation_loss)
            # alpha is the weight of the student loss vs the distillation loss (0 - 1)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            print("Alpha is", self.alpha)
            print("Loss in distiller :", loss)
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`
            self.compiled_metrics.update_state(y, student_prediction)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
            print("Train...", results)
            return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        ## Compute predictions
        y_prediction = self.student(x, training=False)

        # calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        print("Y is:", y)
        print("Y_predicted is",y_prediction)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        print("Test...", results)
        return results

    def call(self, inputs):
        # Pass the input through the teacher and student models
        teacher_predictions = self.teacher(inputs)
        student_predictions = self.student(inputs)
        return teacher_predictions, student_predictions

if train_student_with_fitnet:
    print("TRAINING STUDENT WITH FITNET KD") 
    def objective(trial):
        print("OBJECTIVE NOT DEFINED YET")
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234)",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.5

        #Test
        define_intermediate_networks()

        # Initialize and compile distiller
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        # Distill teacher to student
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student
        student_fitnet.save(directory+student_fitnet._name)



        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename=student_fitnet._name+".tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        if not os.path.isfile(path_student):
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+fold_model_filename, "wb") as output_file:
                output_file.write(tflite_model)

#------------RELATION AND RESPONSE KD WITH STAGE-WISE TRAINING------------------

class Distiller_relation(tf.keras.Model):
    def __init__(self, student, teacher,alpha_rel):
        super(Distiller_relation, self).__init__()
        self.teacher = teacher
        self.student = student
        self.alpha_rel = alpha_rel
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
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
        super(Distiller_relation, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        print("Teacher predicted")
        print(teacher_predictions)
        print("Alpha is "+str(self.alpha))
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            """ distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) """

            s = tf.nn.l2_normalize(student_predictions, 1)
            t = tf.nn.l2_normalize(teacher_predictions, 1)

            distance_loss = Huber_loss(Distance_wise_potential(s), Distance_wise_potential(t))
            angle_loss = Huber_loss(Angle_wise_potential(s), Angle_wise_potential(t))

            distillation_loss = (distance_loss + angle_loss) * 20
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

class Distiller_response(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_response, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
    
        super(Distiller_response, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha


    def train_step(self, data):
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_prediction)


            # Distillation loss for response-based knowledge
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_prediction / self.temperature, axis=1),
                tf.nn.softmax(student_prediction / self.temperature, axis=1)
            )
            """ distillation_loss = self.distillation_loss_fn(
                teacher_prediction / self.temperature,
                student_prediction / self.temperature
                ) """
            print("Distillation loss is")
            print(distillation_loss)
            # alpha is the weight of the student loss vs the distillation loss (0 - 1)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            print("Alpha is", self.alpha)
            print("Loss in distiller :", loss)
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`
            self.compiled_metrics.update_state(y, student_prediction)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
            print("Train...", results)
            return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        ## Compute predictions
        y_prediction = self.student(x, training=False)

        # calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        print("Y is:", y)
        print("Y_predicted is",y_prediction)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        print("Test...", results)
        return results

    def call(self, inputs):
        # Pass the input through the teacher and student models
        teacher_predictions = self.teacher(inputs)
        student_predictions = self.student(inputs)
        return teacher_predictions, student_predictions

if train_student_with_relational_resp_v2_kd:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "student_rel_resp_v2_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)
        # Initialize and compile distiller
        distiller = Distiller_relation(student=student_rel_kd, teacher=teacher,alpha_rel=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_kd,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=student_rel_kd, teacher=teacher,alpha=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_kd,
            temperature=3,
         )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        student_rel_kd = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rel_resp_v2_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rel_resp_v2_kd'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_rel_resp_v2_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_rel_resp_v2_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rel_resp_v2_kd")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in relation-based with response kd and stage-wise training with AI4EDGE_4 ",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "student_rel_resp_v2_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = 0.565
        # Initialize and compile distiller
        distiller = Distiller_relation(student=student_rel_kd, teacher=teacher,alpha_rel=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_kd,
            temperature=3,
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        for single_epoch in range(int(epocas/2)):

            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])

        print("Second stage of training")
        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_rel_kd, teacher=teacher,alpha=alpha_rel_kd)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
            metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=alpha_rel_kd,
            temperature=3,
         )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd")

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rel_resp_v2_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        if not os.path.isfile(path_student):
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_rel_resp_v2_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)

print("----------------TESTING-------------------")

print("Testing teacher")
teacher.evaluate(X_validation,Y_validation)

print("Testing initial student")
student.evaluate(X_validation,Y_validation)

""" if os.path.isdir(directory+"student_rel_kd"):
    print("Testing relational-based student")
    student_rel_kd=loadModelH5(directory+"student_rel_kd")
    student_rel_kd.evaluate(X_validation,Y_validation) """

""" if os.path.isdir(directory+"student_rel_resp_kd"):
    print("Testing relational and response based student")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_kd")
    student_rel_kd.evaluate(X_validation,Y_validation) """

if os.path.isdir(directory+"student_fitnet_kd"):

    print("Testing Fitnet based student")
    student_rel_kd=loadModelH5(directory+"student_fitnet_kd")
    student_rel_kd.evaluate(X_validation,Y_validation)

""" if os.path.isdir(directory+"student_rel_resp_v2_kd"):

    print("Testing relational and response based student with stagewise")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_v2_kd")
    student_rel_kd.evaluate(X_validation,Y_validation) """

""" if os.path.isdir(directory+"student_rel_resp_kd"):

    print("Testing relational and response based student")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_kd")
    student_rel_kd.evaluate(X_validation,Y_validation) """
