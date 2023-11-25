import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ['TF_ENABLE_ONEDNN_OPTS']="1"
import h5py
import math

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


dasetName="kerasTuner8k_140_8_WavSpecAug"
hdf5_path_Train = current_dir+"/"+"data/8k_140_8/WavSpecAug/Train_fs8000_140_2048_0.256_0.128.h5"
hdf5_path_Test = current_dir+"/"+"data/8k_140_8/WavSpecAug/Test_fs8000_140_2048_0.256_0.128.h5"


directory=current_dir+"/"+"resultados/AI4EDGE_4/"
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
    with h5py.File(hdf5_path_Test, 'r') as hf:
        print(hf.keys())

    return X_train,X_validation,Y_validation,Y_train

#epocas = 200
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

# Concatenate dados treino e validação
X_train_and_val = np.concatenate((X_train, X_validation))
#Y_train_and_val = np.concatenate((Y_train, Y_validation))

del X_Test, Y_Test

Y_train = labelsEncoding('Train', scene_labels, Y_train)
Y_validation = labelsEncoding('Val', scene_labels, Y_validation)

Y_train_and_val = np.concatenate((Y_train, Y_validation))


X_train = np.expand_dims(X_train, -1)
Y_train = np.expand_dims(Y_train, -1)

X_validation = np.expand_dims(X_validation, -1)
Y_validation = np.expand_dims(Y_validation, -1)

X_train_and_val = np.expand_dims(X_train_and_val, -1)
Y_train_and_val = np.expand_dims(Y_train_and_val, -1)

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
        print("Alpha is", self.alpha)
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


#load student model

#load teacher model

#test teacher model

#test student model

print("Program done")



#---------------------ADDED SECTION (CARLOS)-------------------------
import optuna
import tensorflow_model_optimization as tfmot

#-----------STUDENT-TRAIN--------------
train_student_with_relational_kd=False
train_student_with_relational_response_kd=False

train_student_with_response_kd=False

train_student_w_rrs_KD=False
train_student_w_rrs_KD_qat=True
train_student_w_rrs_KD_w_train_and_val_data=False
train_student_w_rrs_KD_more_filters=False
train_student_w_rrs_KD_more_layers=False
train_std_w_rrs_mo_layers_mo_dropout=False
test_rrs_qat=False


train_student_w_rrs_KD_residual_con=False
train_student_w_rrs_KD_residual_con_v2=False
train_student_w_rrs_KD_residual_con_deeper=False

train_student_with_fitnet=False
train_student_with_fitnet_deeper_layers=False
train_student_with_fitnet_deeper_v2_layers=False

train_student_with_fitnet_second_t=False
train_student_with_fitnet_second_t_deeper=False
train_student_with_fitnet_second_t_earlier=False

train_student_with_cofd=False
train_student_with_cofd_deeper_layers=False
train_student_with_cofd_second_t=False

train_std_wo_kd=False

#--------SECONDARY TEACHER TRAIN-------

train_teacher_with_relational_resp_v2_kd=False
train_teacher_with_relational_kd=False
train_second_teacher_without_kd=False

train_sec_teacher_with_rrs_res_con_kd=False
train_sec_teacher_with_rrs_res_con_deeper_kd=False

#--------PRUNING-----------------
train_std_w_rrs_mo_layers_pruning=False

second_teacher_prune=False
train_std_w_rrs_mo_layers_pruning_second=False
train_std_w_rel_pruning_second=False


optuna_trials = 50
train_optuna=True
shuffle_training_data=False


#---------------RESPONSE-BASED KD-----------------------

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
                tf.nn.softmax(teacher_prediction, axis=1),
                tf.nn.softmax(student_prediction, axis=1)
            )
            distillation_loss = distillation_loss*50
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


if train_student_with_response_kd:
    print("TRAINING STUDENT WITH RESPONSE KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "Student_resp_kd_test"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0, 1)
        # Initialize and compile distiller
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

        fold_model_filename="Student_resp_kd_test.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['Student_resp_kd_test'],
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
            (df['Network'] == 'Student_resp_kd_test') & (
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

            with open(directory+"Student_resp_kd_test.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"Student_resp_kd_test")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in response-based kd with AI4EDGE_4",
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


def prune_filters(model,input_Shape):

    new_model = BM2_slightly_more_layers_bigger_test(input_Shape)

    for layer_index, layer in enumerate(new_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Get the weights of the convolutional layer
            model.summary()
            print(model.layers[layer_index]._name)
            if isinstance(model.layers[layer_index], tf.keras.layers.Conv2D):
                weights,biases = model.layers[layer_index].get_weights()
            else:
                weights,biases = model.layers[layer_index+1].get_weights()
            # Compute the L1-norm of each filter
            #filter_norms = tf.norm(weights, ord=1, axis=[0, 1, 2])
            filter_norms = tf.reduce_sum(tf.abs(weights), axis=[0, 1, 2])

            # Get the indices of the filters with the smallest L1-norm
            #smallest_filters = tf.argsort(filter_norms)[:num_filters_to_prune]

            # Create a mask to zero out the smallest filters
            #mask = np.ones(15, dtype=bool)
            #mask[smallest_filters] = False

            largest_filters=tf.argsort(filter_norms)[-new_model.layers[layer_index].filters:]
            mask = np.zeros(weights.shape[3], dtype=bool)
            mask[largest_filters] = True

            # Update the weights of the convolutional layer
            new_weights = weights[:, :, :new_model.layers[layer_index].input_shape[-1], mask]
            new_biases =  biases[mask]
            
            layer.set_weights([new_weights,new_biases])
            # Update the output shape of the convolutional layer
            #layer._batch_input_shape = (None, *layer.output_shape[1:])

            #layer.filters = new_weights.shape[-1]

            
    new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return new_model


if train_std_w_rel_pruning_second:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_layers_bigger(input_Shape)
        student_rel_kd._name = "student_rel_pruning_second"
        student_rel_kd.summary()

        
        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]

        second_teacher=loadModelH5(directory+"second_teacher_w_rel_kd")
        second_teacher.fit(X_train, Y_train, epochs=int(1), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        student_rel_kd = prune_filters(second_teacher,input_Shape)
        student_rel_kd._name = "student_rel_pruning_second"

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

        fold_model_filename="student_rel_pruning_second.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        student_rel_kd.summary()


        data = {
                'Network': ['student_rel_pruning_second'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        #print("Convertendo student")
        #converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        #converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        #converter.representative_dataset = representative_dataset
        #tflite_model = converter.convert()

        #with open(directory+"student_rrs_more_layers_pruning_test_2.tflite", "wb") as output_file:
        #    output_file.write(tflite_model)
            
        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_rel_pruning_second') & (
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

            with open(directory+"student_rel_pruning_second.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rel_pruning_second")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in Rel KD with AI4EDGE_4 PRUNING From second teacher",
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
def define_intermediate_networks(layer_until_name, student):
    global teacher_concatenated, student_intermediate
    layer_name = layer_until_name

    teacher0=loadModelH5(directory+"keras_model_0.h5")
    teacher1=loadModelH5(directory+"keras_model_1.h5")
    teacher2=loadModelH5(directory+"keras_model_2.h5")
    teacher3=loadModelH5(directory+"keras_model_3.h5")
    teacher4=loadModelH5(directory+"keras_model_4.h5")
    teacher5=loadModelH5(directory+"keras_model_5.h5")
    teacher6=loadModelH5(directory+"keras_model_6.h5")
    teacher7=loadModelH5(directory+"keras_model_7.h5")
    teacher8=loadModelH5(directory+"keras_model_8.h5")
    teacher9=loadModelH5(directory+"keras_model_9.h5")

    teacher_0_intermediate = tf.keras.Model(inputs=teacher0.input,outputs=teacher0.get_layer(layer_name).output)
    teacher_0_intermediate._name = "teacher_0_intermediate"
    teacher_0_intermediate.summary()

    teacher_1_intermediate = tf.keras.Model(inputs=teacher1.input,outputs=teacher1.get_layer(layer_name).output)
    teacher_1_intermediate._name = "teacher_1_intermediate"

    teacher_2_intermediate = tf.keras.Model(inputs=teacher2.input,outputs=teacher2.get_layer(layer_name).output)
    teacher_2_intermediate._name = "teacher_2_intermediate"

    teacher_3_intermediate = tf.keras.Model(inputs=teacher3.input,outputs=teacher3.get_layer(layer_name).output)
    teacher_3_intermediate._name = "teacher_3_intermediate"

    teacher_4_intermediate = tf.keras.Model(inputs=teacher4.input,outputs=teacher4.get_layer(layer_name).output)
    teacher_4_intermediate._name = "teacher_4_intermediate"

    teacher_5_intermediate = tf.keras.Model(inputs=teacher5.input,outputs=teacher5.get_layer(layer_name).output)
    teacher_5_intermediate._name = "teacher_5_intermediate"

    teacher_6_intermediate = tf.keras.Model(inputs=teacher6.input,outputs=teacher6.get_layer(layer_name).output)
    teacher_6_intermediate._name = "teacher_6_intermediate"

    teacher_7_intermediate = tf.keras.Model(inputs=teacher7.input,outputs=teacher7.get_layer(layer_name).output)
    teacher_7_intermediate._name = "teacher_7_intermediate"

    teacher_8_intermediate = tf.keras.Model(inputs=teacher8.input,outputs=teacher8.get_layer(layer_name).output)
    teacher_8_intermediate._name = "teacher_8_intermediate"

    teacher_9_intermediate = tf.keras.Model(inputs=teacher9.input,outputs=teacher9.get_layer(layer_name).output)
    teacher_9_intermediate._name = "teacher_9_intermediate"



    """ teacher_concatenated_output = tf.keras.layers.Concatenate()(
                             [teacher_0_intermediate.output,
                              teacher_1_intermediate.output,
                              teacher_2_intermediate.output,
                              teacher_3_intermediate.output,
                              teacher_4_intermediate.output,
                              teacher_5_intermediate.output,
                              teacher_6_intermediate.output,
                              teacher_7_intermediate.output,
                              teacher_8_intermediate.output,
                              teacher_9_intermediate.output])
    
    teacher_concatenated = tf.keras.Model(inputs=teacher_0_intermediate.input,outputs=teacher_concatenated_output) """
    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    input = tf.keras.layers.Input(shape=input_Shape)
    t0=teacher_0_intermediate(input,training=False)
    t1=teacher_1_intermediate(input,training=False)
    t2=teacher_2_intermediate(input,training=False)
    t3=teacher_3_intermediate(input,training=False)
    t4=teacher_4_intermediate(input,training=False)
    t5=teacher_5_intermediate(input,training=False)
    t6=teacher_6_intermediate(input,training=False)
    t7=teacher_7_intermediate(input,training=False)
    t8=teacher_8_intermediate(input,training=False)
    t9=teacher_9_intermediate(input,training=False)

    concat = tf.keras.layers.Concatenate()([t0,t1,t2,t3,t4,t5,t6,t7,t8,t9])
    teacher_concatenated = tf.keras.Model(input, concat)

    """ x0=model0(input,training=False)
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
    output = tf.keras.layers.Dense(10, activation='softmax')(concat) """

    teacher_concatenated._name="teacher_concatenated"
    teacher_concatenated.summary()

    student_intermediate = tf.keras.Model(inputs=student.input,outputs=student.get_layer(layer_name).output)
    student_intermediate._name = "student_intermediate"
    student_intermediate.summary()
    

    """ 
    for i, layer in enumerate(teacher0.layers):
        if i <= 3:  
            print("Layer of teacher is ",layer.name)
            print("Layer of teacher intermd is ",teacher_0_intermediate.layers[i-1].name)
            
            teacher_0_intermediate.layers[i-1].set_weights(layer.get_weights())

    teacher_0_intermediate.trainable = False
    teacher_0_intermediate.summary() """
    del teacher0,teacher1,teacher2,teacher3,teacher4,teacher5,teacher6,teacher7,teacher8,teacher9,
    teacher_0_intermediate,teacher_1_intermediate,teacher_2_intermediate,teacher_3_intermediate,teacher_4_intermediate,teacher_5_intermediate,teacher_6_intermediate,
    teacher_7_intermediate,teacher_8_intermediate,teacher_9_intermediate,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9

    print("Teachers loaded")

class Distiller_fitnet(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_fitnet, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """ self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_0_intermediate.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])
        self.student_aux_layers_v3._name="student_aux_v3" """
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

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            #aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

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
                tf.nn.softmax(teacher_prediction, axis=1),
                tf.nn.softmax(student_prediction, axis=1)
            )
            distillation_loss = distillation_loss*50
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

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, student_in,layer_until_in):
        super(CustomCallback, self).__init__()
        self.student = student_in
        self.layer_until=layer_until_in
    def on_epoch_end(self, epoch, logs=None):
        student = self.student
        student_intermediate = tf.keras.Model(inputs=student.input,outputs=student.get_layer(self.layer_until).output)
        student_intermediate._name = "student_intermediate"
        #student_intermediate.summary()

if train_student_with_fitnet:
    print("TRAINING STUDENT WITH FITNET KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-2'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #Second stage of training

        alpha_rel_kd = trial.suggest_float('alpha_kd', 0,1)
        #alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_fitnet_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_fitnet_kd'],
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
            (df['Network'] == 'student_fitnet_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_fitnet_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_fitnet_kd")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234) for AI4EDGE4",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-2'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        


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


#---------------FITNET KD ON DEEPER LAYERS-----------------------
def define_intermediate_networks(layer_until_name, student):
    global teacher_concatenated, student_intermediate
    layer_name = layer_until_name

    teacher0=loadModelH5(directory+"keras_model_0.h5")
    teacher0.summary()
    teacher1=loadModelH5(directory+"keras_model_1.h5")
    teacher2=loadModelH5(directory+"keras_model_2.h5")
    teacher3=loadModelH5(directory+"keras_model_3.h5")
    teacher4=loadModelH5(directory+"keras_model_4.h5")
    teacher5=loadModelH5(directory+"keras_model_5.h5")
    teacher6=loadModelH5(directory+"keras_model_6.h5")
    teacher7=loadModelH5(directory+"keras_model_7.h5")
    teacher8=loadModelH5(directory+"keras_model_8.h5")
    teacher9=loadModelH5(directory+"keras_model_9.h5")

    teacher_0_intermediate = tf.keras.Model(inputs=teacher0.input,outputs=teacher0.get_layer(layer_name).output)
    teacher_0_intermediate._name = "teacher_0_intermediate"
    teacher_0_intermediate.summary()

    teacher_1_intermediate = tf.keras.Model(inputs=teacher1.input,outputs=teacher1.get_layer(layer_name).output)
    teacher_1_intermediate._name = "teacher_1_intermediate"

    teacher_2_intermediate = tf.keras.Model(inputs=teacher2.input,outputs=teacher2.get_layer(layer_name).output)
    teacher_2_intermediate._name = "teacher_2_intermediate"

    teacher_3_intermediate = tf.keras.Model(inputs=teacher3.input,outputs=teacher3.get_layer(layer_name).output)
    teacher_3_intermediate._name = "teacher_3_intermediate"

    teacher_4_intermediate = tf.keras.Model(inputs=teacher4.input,outputs=teacher4.get_layer(layer_name).output)
    teacher_4_intermediate._name = "teacher_4_intermediate"

    teacher_5_intermediate = tf.keras.Model(inputs=teacher5.input,outputs=teacher5.get_layer(layer_name).output)
    teacher_5_intermediate._name = "teacher_5_intermediate"

    teacher_6_intermediate = tf.keras.Model(inputs=teacher6.input,outputs=teacher6.get_layer(layer_name).output)
    teacher_6_intermediate._name = "teacher_6_intermediate"

    teacher_7_intermediate = tf.keras.Model(inputs=teacher7.input,outputs=teacher7.get_layer(layer_name).output)
    teacher_7_intermediate._name = "teacher_7_intermediate"

    teacher_8_intermediate = tf.keras.Model(inputs=teacher8.input,outputs=teacher8.get_layer(layer_name).output)
    teacher_8_intermediate._name = "teacher_8_intermediate"

    teacher_9_intermediate = tf.keras.Model(inputs=teacher9.input,outputs=teacher9.get_layer(layer_name).output)
    teacher_9_intermediate._name = "teacher_9_intermediate"



    """ teacher_concatenated_output = tf.keras.layers.Concatenate()(
                             [teacher_0_intermediate.output,
                              teacher_1_intermediate.output,
                              teacher_2_intermediate.output,
                              teacher_3_intermediate.output,
                              teacher_4_intermediate.output,
                              teacher_5_intermediate.output,
                              teacher_6_intermediate.output,
                              teacher_7_intermediate.output,
                              teacher_8_intermediate.output,
                              teacher_9_intermediate.output])
    
    teacher_concatenated = tf.keras.Model(inputs=teacher_0_intermediate.input,outputs=teacher_concatenated_output) """
    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    input = tf.keras.layers.Input(shape=input_Shape)
    t0=teacher_0_intermediate(input,training=False)
    t1=teacher_1_intermediate(input,training=False)
    t2=teacher_2_intermediate(input,training=False)
    t3=teacher_3_intermediate(input,training=False)
    t4=teacher_4_intermediate(input,training=False)
    t5=teacher_5_intermediate(input,training=False)
    t6=teacher_6_intermediate(input,training=False)
    t7=teacher_7_intermediate(input,training=False)
    t8=teacher_8_intermediate(input,training=False)
    t9=teacher_9_intermediate(input,training=False)

    concat = tf.keras.layers.Concatenate()([t0,t1,t2,t3,t4,t5,t6,t7,t8,t9])
    teacher_concatenated = tf.keras.Model(input, concat)

    """ x0=model0(input,training=False)
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
    output = tf.keras.layers.Dense(10, activation='softmax')(concat) """

    teacher_concatenated._name="teacher_concatenated"
    teacher_concatenated.summary()

    student_intermediate = tf.keras.Model(inputs=student.input,outputs=student.get_layer(layer_name).output)
    student_intermediate._name = "student_intermediate"
    student_intermediate.summary()
    

    """ 
    for i, layer in enumerate(teacher0.layers):
        if i <= 3:  
            print("Layer of teacher is ",layer.name)
            print("Layer of teacher intermd is ",teacher_0_intermediate.layers[i-1].name)
            
            teacher_0_intermediate.layers[i-1].set_weights(layer.get_weights())

    teacher_0_intermediate.trainable = False
    teacher_0_intermediate.summary() """
    del teacher0,teacher1,teacher2,teacher3,teacher4,teacher5,teacher6,teacher7,teacher8,teacher9,
    teacher_0_intermediate,teacher_1_intermediate,teacher_2_intermediate,teacher_3_intermediate,teacher_4_intermediate,teacher_5_intermediate,teacher_6_intermediate,
    teacher_7_intermediate,teacher_8_intermediate,teacher_9_intermediate,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9

    print("Teachers loaded")

class Distiller_fitnet(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_fitnet, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """ self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_0_intermediate.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])
        self.student_aux_layers_v3._name="student_aux_v3" """
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

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            #aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

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
                tf.nn.softmax(teacher_prediction, axis=1),
                tf.nn.softmax(student_prediction, axis=1)
            )
            distillation_loss = distillation_loss*50
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

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, student_in,layer_until_in):
        super(CustomCallback, self).__init__()
        self.student = student_in
        self.layer_until=layer_until_in
    def on_epoch_end(self, epoch, logs=None):
        student = self.student
        student_intermediate = tf.keras.Model(inputs=student.input,outputs=student.get_layer(self.layer_until).output)
        student_intermediate._name = "student_intermediate"
        #student_intermediate.summary()

if train_student_with_fitnet_deeper_layers:
    print("TRAINING STUDENT WITH FITNET KD DEEPER") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_deeper_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #Second stage of training

        alpha_rel_kd = trial.suggest_float('alpha_kd', 0, 0.01)
        #alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_fitnet_deeper_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_fitnet_deeper_kd'],
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
            (df['Network'] == 'student_fitnet_deeper_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_fitnet_deeper_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_fitnet_deeper_kd")

        acc_stud = metrics_stud[1]
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234) for AI4EDGE4 DEEPER LAYERS",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        


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


if train_student_with_fitnet_deeper_v2_layers:
    print("TRAINING STUDENT WITH FITNET KD DEEPER") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_deeper_v2_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3_0'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #Second stage of training

        alpha_rel_kd = trial.suggest_float('alpha_kd', 0, 0.01)
        #alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_fitnet_deeper_v2_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_fitnet_deeper_v2_kd'],
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
            (df['Network'] == 'student_fitnet_deeper_v2_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_fitnet_deeper_v2_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_fitnet_deeper_v2_kd")

        acc_stud = metrics_stud[1]
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234) for AI4EDGE4 DEEPER LAYERS v2",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)


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

if train_student_w_rrs_KD:
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
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

quantize_model = tfmot.quantization.keras.quantize_model


if train_student_w_rrs_KD_qat:
    print("TRAINING STUDENT WITH RRS KD and QAT") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_qat(input_Shape)
        student_rel_kd._name = "student_rrs_kd_qat"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)
        # Pass the CustomQuantizeConfig class to the quantize_scope argument
        with tfmot.quantization.keras.quantize_scope({'CustomQuantizeConfig': CustomQuantizeConfig}):
            # q_aware stands for for quantization aware.
            student_rel_kd = quantize_model(student_rel_kd)
        
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

        student_rel_kd.summary()


        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rrs_kd_qat.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        student_rel_kd.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['categorical_accuracy']
        )
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_kd_qat'],
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
            (df['Network'] == 'student_rrs_kd_qat') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            quantized_tflite_model = converter.convert()


            with open(directory+"student_rrs_kd_qat.tflite", "wb") as output_file:
                output_file.write(quantized_tflite_model)
            
            #student_rel_kd.save(directory+"student_rrs_kd_qat")
            student_rel_kd.save_weights(directory + "student_rrs_kd_qat_weights.h5")
            # Save the model's weights
            #

            # Recreate the model architecture
            #new_student_rel_kd = BM2_qat(input_Shape)
            #new_student_rel_kd._name = "student_rrs_kd_qat"

            # Load the saved weights into the new model
            #new_student_rel_kd.load_weights(directory + "student_rrs_kd_qat_weights.h5")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha RRS KD training with QAT in AI4EDGE_4 ",
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


if train_student_w_rrs_KD_w_train_and_val_data:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        
        input_Shape = (X_train_and_val.shape[1], X_train_and_val.shape[2], 1)
        student_rel_kd = BM2(input_Shape)
        student_rel_kd._name = "student_rrs_full_data_kd"
        student_rel_kd.summary()
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
        history = distiller.fit(X_train_and_val, Y_train_and_val, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train_and_val, Y_train_and_val, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_rel_kd = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train_and_val.shape[0], size=round((X_train_and_val.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train_and_val
        representative_data = X_train_and_val[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rrs_full_data_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_full_data_kd'],
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
            (df['Network'] == 'student_rrs_full_data_kd') & (
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

            with open(directory+"student_rrs_full_data_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_full_data_kd")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS training with AI4EDGE_4 and train + val",
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

if train_student_w_rrs_KD_more_filters:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_filters(input_Shape)
        student_rel_kd._name = "student_rrs_more_filters"
        student_rel_kd.summary()
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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

        fold_model_filename="student_rrs_more_filters.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_more_filters'],
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
            (df['Network'] == 'student_rrs_more_filters') & (
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

            with open(directory+"student_rrs_more_filters.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_more_filters")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS with AI4EDGE_4 (MORE FILTERS)",
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

if train_student_w_rrs_KD_more_layers:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_layers(input_Shape)
        student_rel_kd._name = "student_rrs_more_layers_v2"
        student_rel_kd.summary()
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
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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

        fold_model_filename="student_rrs_more_layers_v2.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_more_layers_v2'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        
        #with open(directory+"student_rrs_more_layers_v2.tflite", "wb") as output_file:
        #    output_file.write(tflite_model)
            
        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_rrs_more_layers_v2') & (
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

            with open(directory+"student_rrs_more_layers_v2.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_more_layers_v2")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS with AI4EDGE_4 (MORE LAYERS) v2",
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


if train_std_w_rrs_mo_layers_mo_dropout:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_layers_more_drop(input_Shape)
        student_rel_kd._name = "student_rrs_more_layers_more_drop"
        student_rel_kd.summary()
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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

        fold_model_filename="student_rrs_more_layers_more_drop.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_more_layers_more_drop'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        print("Convertendo student")
        converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        converter.representative_dataset = representative_dataset
        tflite_model = converter.convert()

        with open(directory+"student_rrs_more_layers_more_drop.tflite", "wb") as output_file:
            output_file.write(tflite_model)
            
        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_rrs_more_layers_more_drop') & (
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

            with open(directory+"student_rrs_more_layers_more_drop.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_more_layers_more_drop")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS with AI4EDGE_4 (MORE LAYERS) MORE DROPOUT",
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

if train_std_w_rrs_mo_layers_pruning:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_layers_bigger(input_Shape)
        student_rel_kd._name = "student_rrs_more_layers_pruning_test_3"
        student_rel_kd.summary()
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0, 1)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        #callback_list = [
        #    dcase_util.tfkeras.StasherCallback(
        #        epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        #    ),
        #    tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
        #    tfmot.sparsity.keras.UpdatePruningStep()
        #]
        #student_rel_kd = distiller.student

        #num_examples=X_train.shape[0]

        #end_step= np.ceil(1.0*num_examples/batch_size).astype(np.int32)*epocas
        #print("End step is:")
        #print(end_step)
        #pruning_params = { 'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.1,
        #final_sparsity=0.50,
        #begin_step=2,
        #end_step=end_step)}

        #model_pruning=tfmot.sparsity.keras.prune_low_magnitude(student_rel_kd, **pruning_params)
        
        #model_pruning.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
        #model_pruning.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #student_rel_kd=tfmot.sparsity.keras.strip_pruning(model_pruning)
        #student_rel_kd=model_pruning
        def prune_filters(model):

            new_model = BM2_slightly_more_layers_bigger_test(input_Shape)

            for layer_index, layer in enumerate(new_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    # Get the weights of the convolutional layer
                    weights,biases = model.layers[layer_index].get_weights()

                    # Compute the L1-norm of each filter
                    #filter_norms = tf.norm(weights, ord=1, axis=[0, 1, 2])
                    filter_norms = tf.reduce_sum(tf.abs(weights), axis=[0, 1, 2])

                    # Get the indices of the filters with the smallest L1-norm
                    #smallest_filters = tf.argsort(filter_norms)[:num_filters_to_prune]

                    # Create a mask to zero out the smallest filters
                    #mask = np.ones(15, dtype=bool)
                    #mask[smallest_filters] = False

                    largest_filters=tf.argsort(filter_norms)[-new_model.layers[layer_index].filters:]
                    mask = np.zeros(weights.shape[3], dtype=bool)
                    mask[largest_filters] = True

                    # Update the weights of the convolutional layer
                    new_weights = weights[:, :, :new_model.layers[layer_index].input_shape[-1], mask]
                    new_biases =  biases[mask]
                    
                    layer.set_weights([new_weights,new_biases])
                    # Update the output shape of the convolutional layer
                    #layer._batch_input_shape = (None, *layer.output_shape[1:])

                    #layer.filters = new_weights.shape[-1]

                    
            new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
            return new_model

        student_rel_kd = prune_filters(student_rel_kd)
        student_rel_kd.fit(X_train, Y_train, epochs=int(epocas/2), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
    
        #student_rel_kd.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])


        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rrs_more_layers_pruning_test_3.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        student_rel_kd.summary()


        data = {
                'Network': ['student_rrs_more_layers_pruning_test_3'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        #print("Convertendo student")
        #converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        #converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        #converter.representative_dataset = representative_dataset
        #tflite_model = converter.convert()

        #with open(directory+"student_rrs_more_layers_pruning_test_2.tflite", "wb") as output_file:
        #    output_file.write(tflite_model)
            
        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_rrs_more_layers_pruning_test_3') & (
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

            with open(directory+"student_rrs_more_layers_pruning_test_3.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_more_layers_pruning_test_3")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS with AI4EDGE_4 (MORE LAYERS) PRUNING",
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


def prune_filters(model,input_Shape):

    new_model = BM2_slightly_more_layers_bigger_test(input_Shape)

    for layer_index, layer in enumerate(new_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Get the weights of the convolutional layer
            model.summary()
            print(model.layers[layer_index]._name)
            if isinstance(model.layers[layer_index], tf.keras.layers.Conv2D):
                weights,biases = model.layers[layer_index].get_weights()
            else:
                weights,biases = model.layers[layer_index+1].get_weights()
            # Compute the L1-norm of each filter
            #filter_norms = tf.norm(weights, ord=1, axis=[0, 1, 2])
            filter_norms = tf.reduce_sum(tf.abs(weights), axis=[0, 1, 2])

            # Get the indices of the filters with the smallest L1-norm
            #smallest_filters = tf.argsort(filter_norms)[:num_filters_to_prune]

            # Create a mask to zero out the smallest filters
            #mask = np.ones(15, dtype=bool)
            #mask[smallest_filters] = False

            largest_filters=tf.argsort(filter_norms)[-new_model.layers[layer_index].filters:]
            mask = np.zeros(weights.shape[3], dtype=bool)
            mask[largest_filters] = True

            # Update the weights of the convolutional layer
            new_weights = weights[:, :, :new_model.layers[layer_index].input_shape[-1], mask]
            new_biases =  biases[mask]
            
            layer.set_weights([new_weights,new_biases])
            # Update the output shape of the convolutional layer
            #layer._batch_input_shape = (None, *layer.output_shape[1:])

            #layer.filters = new_weights.shape[-1]

            
    new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return new_model

if train_std_w_rrs_mo_layers_pruning_second:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_layers_bigger(input_Shape)
        student_rel_kd._name = "student_rrs_pruning_second"
        student_rel_kd.summary()

        
        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]

        second_teacher=loadModelH5(directory+"second_teacher_w_rel_kd")
        second_teacher.fit(X_train, Y_train, epochs=int(1), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        student_rel_kd = prune_filters(second_teacher,input_Shape)
        student_rel_kd._name = "student_rrs_pruning_second"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0, 1)
        # Initialize and compile distiller
        distiller = Distiller_relation(student=student_rel_kd, teacher=second_teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=student_rel_kd, teacher=second_teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        #callback_list = [
        #    dcase_util.tfkeras.StasherCallback(
        #        epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        #    ),
        #    tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
        #    tfmot.sparsity.keras.UpdatePruningStep()
        #]
        #student_rel_kd = distiller.student

        #num_examples=X_train.shape[0]

        #end_step= np.ceil(1.0*num_examples/batch_size).astype(np.int32)*epocas
        #print("End step is:")
        #print(end_step)
        #pruning_params = { 'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.1,
        #final_sparsity=0.50,
        #begin_step=2,
        #end_step=end_step)}

        #model_pruning=tfmot.sparsity.keras.prune_low_magnitude(student_rel_kd, **pruning_params)
        
        #model_pruning.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
        
        #model_pruning.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #student_rel_kd=tfmot.sparsity.keras.strip_pruning(model_pruning)
        #student_rel_kd=model_pruning



        # def prune_filters(model):

        #     new_model = BM2_slightly_more_layers_bigger_test(input_Shape)

        #     for layer_index, layer in enumerate(new_model.layers):
        #         if isinstance(layer, tf.keras.layers.Conv2D):
        #             # Get the weights of the convolutional layer
        #             weights,biases = model.layers[layer_index].get_weights()

        #             # Compute the L1-norm of each filter
        #             #filter_norms = tf.norm(weights, ord=1, axis=[0, 1, 2])
        #             filter_norms = tf.reduce_sum(tf.abs(weights), axis=[0, 1, 2])

        #             # Get the indices of the filters with the smallest L1-norm
        #             #smallest_filters = tf.argsort(filter_norms)[:num_filters_to_prune]

        #             # Create a mask to zero out the smallest filters
        #             #mask = np.ones(15, dtype=bool)
        #             #mask[smallest_filters] = False

        #             largest_filters=tf.argsort(filter_norms)[-new_model.layers[layer_index].filters:]
        #             mask = np.zeros(weights.shape[3], dtype=bool)
        #             mask[largest_filters] = True

        #             # Update the weights of the convolutional layer
        #             new_weights = weights[:, :, :new_model.layers[layer_index].input_shape[-1], mask]
        #             new_biases =  biases[mask]
                    
        #             layer.set_weights([new_weights,new_biases])
        #             # Update the output shape of the convolutional layer
        #             #layer._batch_input_shape = (None, *layer.output_shape[1:])

        #             #layer.filters = new_weights.shape[-1]

                    
        #     new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
        #     return new_model

        # second_teacher=loadModelH5(directory+"second_teacher_w_rel_kd")
        # student_rel_kd = prune_filters(second_teacher)


        # student_rel_kd.fit(X_train, Y_train, epochs=int(epocas/2), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
    
        #student_rel_kd.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])


        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_rrs_pruning_second.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        student_rel_kd.summary()


        data = {
                'Network': ['student_rrs_pruning_second'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        #print("Convertendo student")
        #converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        #converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        #converter.representative_dataset = representative_dataset
        #tflite_model = converter.convert()

        #with open(directory+"student_rrs_more_layers_pruning_test_2.tflite", "wb") as output_file:
        #    output_file.write(tflite_model)
            
        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_rrs_pruning_second') & (
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

            with open(directory+"student_rrs_pruning_second.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_pruning_second")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS with AI4EDGE_4 PRUNING From second teacher",
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


if train_student_w_rrs_KD_residual_con:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_residual(input_Shape)
        student_rel_kd.summary()
        student_rel_kd._name = "student_rrs_res_con_kd"
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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

        fold_model_filename="student_rrs_res_con_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_res_con_kd'],
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
            (df['Network'] == 'student_rrs_res_con_kd') & (
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

            with open(directory+"student_rrs_res_con_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_res_con_kd")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS WITH Residual Connections with AI4EDGE_4 ",
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

#train_student_w_rrs_KD_residual_con_v2

if train_student_w_rrs_KD_residual_con_v2:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_residual_v2(input_Shape)
        student_rel_kd.summary()
        student_rel_kd._name = "student_rrs_res_con_kd_v2"
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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

        fold_model_filename="student_rrs_res_con_kd_v2.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_res_con_kd_v2'],
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
            (df['Network'] == 'student_rrs_res_con_kd_v2') & (
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

            with open(directory+"student_rrs_res_con_kd_v2.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_res_con_kd_v2")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS WITH Residual Connections with AI4EDGE_4 v2",
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


if train_student_w_rrs_KD_residual_con_deeper:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_residual_deeper(input_Shape)
        student_rel_kd.summary()
        student_rel_kd._name = "student_rrs_res_con_kd_deeper"
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

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

        fold_model_filename="student_rrs_res_con_kd_deeper.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_rrs_res_con_kd_deeper'],
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
            (df['Network'] == 'student_rrs_res_con_kd_deeper') & (
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

            with open(directory+"student_rrs_res_con_kd_deeper.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_rrs_res_con_kd_deeper")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in RRS WITH Residual Connections with AI4EDGE_4 deeper",
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




#----------- COMPLETE OVERHAUL OF FEATURE DISTILLATION TRAINING -----------------

class Distiller_cofd(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_cofd, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """ self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_0_intermediate.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])
        self.student_aux_layers_v3._name="student_aux_v3" """
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
        """ Configure the Distiller_cofd.
    student_loss_fn: Loss function of difference between student
                predictions and ground-truth
    distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
    alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
    temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller_cofd, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def get_margin_v2(self,features):

        std_dev = tf.keras.backend.std(features)
        mean = tf.keras.backend.mean(features)
        # margin = np.where(norm.cdf(-mean / std_dev) > 1e-3,
        #                    - std_dev * np.exp(- (mean / std_dev) ** 2 / 2) / np.sqrt(2 * np.pi) / norm.cdf(-mean / std_dev) + mean,-3 * std_dev).astype(np.float32)
        #margin = std_dev + mean
        eps = tf.constant(1e-3, dtype=tf.float32)

        condition = tf.greater(-mean / std_dev, eps)

        tmp1 = - std_dev * tf.exp(- tf.pow(mean / std_dev, 2) / 2) / tf.sqrt(2 * math.pi)
        tmp2 = tf.math.erf(-mean / std_dev)
        tmp3 = -3 * std_dev

        margin = tf.where(condition, tmp1 / tmp2 + mean, tmp3)
        margin_value =  tf.dtypes.cast(margin, tf.float32)
        return margin_value

    def train_step(self, data):

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            #aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

            teach_feat_augmented = aux(teacher_features, training=False)

            margins_teach = self.get_margin_v2(teach_feat_augmented)

            teach_feat_augmented = tf.stop_gradient(tf.maximum(teach_feat_augmented, margins_teach))

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
            print("Loss in Distiller_cofd :", loss)
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

if train_student_with_cofd:
    print("TRAINING STUDENT WITH COFD KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_cofd_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.7, 1)


        #Test
        layer_until='Conv-2'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_cofd(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_cofd_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_cofd_kd'],
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
            (df['Network'] == 'student_cofd_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_cofd_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_cofd_kd")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in cofd kd (seed 1234) for AI4EDGE4",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_cofd_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.55

        #Test
        layer_until='Conv-2'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_cofd(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
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

#----------- COMPLETE OVERHAUL OF FEATURE DISTILLATION TRAINING FOR DEeper layers -----------------

class Distiller_cofd(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_cofd, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """ self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_0_intermediate.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])
        self.student_aux_layers_v3._name="student_aux_v3" """
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
        """ Configure the Distiller_cofd.
    student_loss_fn: Loss function of difference between student
                predictions and ground-truth
    distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
    alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
    temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller_cofd, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def get_margin_v2(self,features):

        std_dev = tf.keras.backend.std(features)
        mean = tf.keras.backend.mean(features)
        # margin = np.where(norm.cdf(-mean / std_dev) > 1e-3,
        #                    - std_dev * np.exp(- (mean / std_dev) ** 2 / 2) / np.sqrt(2 * np.pi) / norm.cdf(-mean / std_dev) + mean,-3 * std_dev).astype(np.float32)
        #margin = std_dev + mean
        eps = tf.constant(1e-3, dtype=tf.float32)

        condition = tf.greater(-mean / std_dev, eps)

        tmp1 = - std_dev * tf.exp(- tf.pow(mean / std_dev, 2) / 2) / tf.sqrt(2 * math.pi)
        tmp2 = tf.math.erf(-mean / std_dev)
        tmp3 = -3 * std_dev

        margin = tf.where(condition, tmp1 / tmp2 + mean, tmp3)
        margin_value =  tf.dtypes.cast(margin, tf.float32)
        return margin_value

    def train_step(self, data):

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            #aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

            teach_feat_augmented = aux(teacher_features, training=False)

            margins_teach = self.get_margin_v2(teach_feat_augmented)

            teach_feat_augmented = tf.stop_gradient(tf.maximum(teach_feat_augmented, margins_teach))

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
            print("Loss in Distiller_cofd :", loss)
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

if train_student_with_cofd_deeper_layers:
    print("TRAINING STUDENT WITH COFD KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_cofd_deeper_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)


        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_cofd(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_cofd_deeper_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_cofd_kd'],
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
            (df['Network'] == 'student_cofd_deeper_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_cofd_deeper_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_cofd_deeper_kd")
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in cofd kd (seed 1234) for AI4EDGE4 FOR DEEPER LAYERS",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_cofd_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.55

        #Test
        layer_until='Conv-2'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_cofd(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
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

#----------------------------------------------------------------------
#----------------------------SECOND-TEACHER----------------------------
#----------------------------------------------------------------------

#------------TRAIN SECONDARY TEACHER WITH RELATION AND RESPONSE KD WITH STAGE-WISE TRAINING------------------

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

if train_teacher_with_relational_resp_v2_kd:
    print("TRAINING Secondary teacher WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher._name = "second_teacher_rel_resp_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)
        # Initialize and compile distiller
        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=second_teacher, teacher=teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        second_teacher = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_rel_resp_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = second_teacher.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['second_teacher_rel_resp_kd'],
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
            (df['Network'] == 'second_teacher_rel_resp_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            """ print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(second_teacher)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"second_teacher_rel_resp_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
             """
            second_teacher.save(directory+"second_teacher_rel_resp_kd")
        del second_teacher
        del student_scratch
        del distiller
        del callback_list
        del history
        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="(secondary_teacher)Changing alpha in relation-based with response kd and stage-wise training with AI4EDGE_4 ",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_rel_resp_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = 0.565
        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=second_teacher, teacher=teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):


            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        second_teacher = distiller.student

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


if train_sec_teacher_with_rrs_res_con_kd:
    print("TRAINING Secondary teacher WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger_residual(input_Shape)
        second_teacher._name = "second_teacher_rrs_res_con_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)
        # Initialize and compile distiller
        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=second_teacher, teacher=teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        second_teacher = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_rrs_res_con_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = second_teacher.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['second_teacher_rrs_res_con_kd'],
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
            (df['Network'] == 'second_teacher_rrs_res_con_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            """ print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(second_teacher)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"second_teacher_rel_resp_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
             """
            second_teacher.save(directory+"second_teacher_rrs_res_con_kd")
        del second_teacher
        del student_scratch
        del distiller
        del callback_list
        del history
        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="(secondary_teacher)Changing alpha in RRS training with AI4EDGE_4 and residual connections",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_rel_resp_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = 0.565
        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=second_teacher, teacher=teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):


            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        second_teacher = distiller.student

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



if train_sec_teacher_with_rrs_res_con_deeper_kd:
    print("TRAINING Secondary teacher WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger_residual_deeper(input_Shape)
        second_teacher._name = "second_teacher_rrs_res_con_DEEPER_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)
        # Initialize and compile distiller
        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=second_teacher, teacher=teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):

            print("Epoch ", (epocas/2 + single_epoch), "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        second_teacher = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_rrs_res_con_DEEPER_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = second_teacher.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['second_teacher_rrs_res_con_DEEPER_kd'],
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
            (df['Network'] == 'second_teacher_rrs_res_con_DEEPER_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            """ print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(second_teacher)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"second_teacher_rel_resp_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
             """
            second_teacher.save(directory+"second_teacher_rrs_res_con_DEEPER_kd")
        del second_teacher
        del student_scratch
        del distiller
        del callback_list
        del history
        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="(secondary_teacher)Changing alpha in RRS training with AI4EDGE_4 and residual connections DEEPER",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_rel_resp_kd"
        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student)
        alpha_rel_kd = 0.565
        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):
            print("Epoch ", single_epoch, "/", epocas)
            # Distill teacher to student
            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k]) """

        print("Second stage of training")
        alpha_rel_kd=0.00001
        distiller = Distiller_response(student=second_teacher, teacher=teacher,alpha=alpha_rel_kd)
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
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        """ for single_epoch in range(int(epocas/2)):


            epoch_history = distiller.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
            for k in history.history:
                history.history[k].extend(epoch_history.history[k])




            student_rel_kd = distiller.student
            student_rel_kd.save(directory+"student_rel_resp_v2_kd") """

        second_teacher = distiller.student

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


#------------TRAIN SECONDARY TEACHER WITHOUT KD------------------
if train_second_teacher_without_kd:
    print("TRAINING Secondary teacher WO KD training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_wo_kd"
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        second_teacher.compile(
            optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy']
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = second_teacher.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_wo_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = second_teacher.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['second_teacher_wo_kd'],
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
            (df['Network'] == 'second_teacher_wo_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            """ print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(second_teacher)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"second_teacher_rel_resp_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
             """
            second_teacher.save(directory+"second_teacher_wo_kd")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="(secondary_teacher) Training wo kd AI4EDGE_4 ",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_wo_kd"
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        second_teacher.compile(
            optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy']
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = second_teacher.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        second_teacher = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_wo_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized


#------------TRAIN SECONDARY TEACHER WITH RELATION KD TRAINING------------------
if train_teacher_with_relational_kd:
    print("TRAINING Secondary teacher W Rel KD training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_w_rel_kd"

        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0, 1)

        distiller = Distiller_relation(student=second_teacher, teacher=teacher,alpha_rel=alpha_rel_kd)

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
        history = second_teacher.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_w_rel_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = second_teacher.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['second_teacher_w_rel_kd'],
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
            (df['Network'] == 'second_teacher_w_rel_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            """ print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(second_teacher)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"second_teacher_rel_resp_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
             """
            second_teacher.save(directory+"second_teacher_w_rel_kd")

        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="(secondary_teacher) Training w REL kd AI4EDGE_4 ",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        second_teacher = BM2_bigger(input_Shape)
        second_teacher.summary()
        second_teacher._name = "second_teacher_wo_kd"
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        second_teacher.compile(
            optimizer=optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy']
        )

        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]
        history = second_teacher.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        second_teacher = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="second_teacher_wo_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized



#------------ SECONDARY TEACHER PRUNING------------------


if second_teacher_prune:
    print("TRAINING STUDENT WITH RELATIONAL AND RESPONSE KD with stage wise training") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_rel_kd = BM2_slightly_more_layers_bigger(input_Shape)
        student_rel_kd._name = "student_prune_second"
        student_rel_kd.summary()

        
        callback_list = [
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True)
        ]

        second_teacher=loadModelH5(directory+"second_teacher_w_rel_kd")
        second_teacher.fit(X_train, Y_train, epochs=int(1), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        student_rel_kd = prune_filters(second_teacher,input_Shape)
        student_rel_kd._name = "student_prune_second"
        student_rel_kd.fit(X_train, Y_train, epochs=int(epocas/4), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)



        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_prune_second.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_rel_kd.evaluate(X_validation,Y_validation)

        student_rel_kd.summary()


        data = {
                'Network': ['student_prune_second'],
                'Testing Accuracy': [metrics_stud[1]],
                'Testing Loss': [metrics_stud[0]],
            }

        # Make data frame of above data
        df = pd.DataFrame(data)

        # append data frame to CSV file
        df.to_csv('Test_data_v2_optuna.csv', mode='a', index=False, header=False)

        # read in the CSV file
        df = pd.read_csv('Test_data_v2_optuna.csv')

        #print("Convertendo student")
        #converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        #converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        #converter.representative_dataset = representative_dataset
        #tflite_model = converter.convert()

        #with open(directory+"student_rrs_more_layers_pruning_test_2.tflite", "wb") as output_file:
        #    output_file.write(tflite_model)
            
        # filter rows where the 'Network' column is 'student_w_r_kd' and 'Accuracy' is greater than 50
        filtered_df = df[
            (df['Network'] == 'student_prune_second') & (
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

            with open(directory+"student_prune_second.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_rel_kd.save(directory+"student_prune_second")

        acc_stud = metrics_stud[1]
        del student_rel_kd
        del callback_list
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="AI4EDGE_4 PRUNING From second teacher",
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





#---------------FITNET KD WITH SECOND TEACHER-----------------------
def define_intermediate_networks_v2(layer_until_name_student,layer_until_name_teacher, student):
    global teacher_concatenated, student_intermediate, second_teacher
    

    """ teacher0=loadModelH5(directory+"keras_model_0.h5")
    teacher0.summary()
    teacher1=loadModelH5(directory+"keras_model_1.h5")
    teacher2=loadModelH5(directory+"keras_model_2.h5")
    teacher3=loadModelH5(directory+"keras_model_3.h5")
    teacher4=loadModelH5(directory+"keras_model_4.h5")
    teacher5=loadModelH5(directory+"keras_model_5.h5")
    teacher6=loadModelH5(directory+"keras_model_6.h5")
    teacher7=loadModelH5(directory+"keras_model_7.h5")
    teacher8=loadModelH5(directory+"keras_model_8.h5")
    teacher9=loadModelH5(directory+"keras_model_9.h5")

    teacher_0_intermediate = tf.keras.Model(inputs=teacher0.input,outputs=teacher0.get_layer(layer_name).output)
    teacher_0_intermediate._name = "teacher_0_intermediate"
    teacher_0_intermediate.summary()

    teacher_1_intermediate = tf.keras.Model(inputs=teacher1.input,outputs=teacher1.get_layer(layer_name).output)
    teacher_1_intermediate._name = "teacher_1_intermediate"

    teacher_2_intermediate = tf.keras.Model(inputs=teacher2.input,outputs=teacher2.get_layer(layer_name).output)
    teacher_2_intermediate._name = "teacher_2_intermediate"

    teacher_3_intermediate = tf.keras.Model(inputs=teacher3.input,outputs=teacher3.get_layer(layer_name).output)
    teacher_3_intermediate._name = "teacher_3_intermediate"

    teacher_4_intermediate = tf.keras.Model(inputs=teacher4.input,outputs=teacher4.get_layer(layer_name).output)
    teacher_4_intermediate._name = "teacher_4_intermediate"

    teacher_5_intermediate = tf.keras.Model(inputs=teacher5.input,outputs=teacher5.get_layer(layer_name).output)
    teacher_5_intermediate._name = "teacher_5_intermediate"

    teacher_6_intermediate = tf.keras.Model(inputs=teacher6.input,outputs=teacher6.get_layer(layer_name).output)
    teacher_6_intermediate._name = "teacher_6_intermediate"

    teacher_7_intermediate = tf.keras.Model(inputs=teacher7.input,outputs=teacher7.get_layer(layer_name).output)
    teacher_7_intermediate._name = "teacher_7_intermediate"

    teacher_8_intermediate = tf.keras.Model(inputs=teacher8.input,outputs=teacher8.get_layer(layer_name).output)
    teacher_8_intermediate._name = "teacher_8_intermediate"

    teacher_9_intermediate = tf.keras.Model(inputs=teacher9.input,outputs=teacher9.get_layer(layer_name).output)
    teacher_9_intermediate._name = "teacher_9_intermediate"



    teacher_concatenated_output = tf.keras.layers.Concatenate()(
                             [teacher_0_intermediate.output,
                              teacher_1_intermediate.output,
                              teacher_2_intermediate.output,
                              teacher_3_intermediate.output,
                              teacher_4_intermediate.output,
                              teacher_5_intermediate.output,
                              teacher_6_intermediate.output,
                              teacher_7_intermediate.output,
                              teacher_8_intermediate.output,
                              teacher_9_intermediate.output])
    
    teacher_concatenated = tf.keras.Model(inputs=teacher_0_intermediate.input,outputs=teacher_concatenated_output)
    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    input = tf.keras.layers.Input(shape=input_Shape)
    t0=teacher_0_intermediate(input,training=False)
    t1=teacher_1_intermediate(input,training=False)
    t2=teacher_2_intermediate(input,training=False)
    t3=teacher_3_intermediate(input,training=False)
    t4=teacher_4_intermediate(input,training=False)
    t5=teacher_5_intermediate(input,training=False)
    t6=teacher_6_intermediate(input,training=False)
    t7=teacher_7_intermediate(input,training=False)
    t8=teacher_8_intermediate(input,training=False)
    t9=teacher_9_intermediate(input,training=False)

    concat = tf.keras.layers.Concatenate()([t0,t1,t2,t3,t4,t5,t6,t7,t8,t9])
    teacher_concatenated = tf.keras.Model(input, concat)

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

    teacher_concatenated._name="teacher_concatenated"
    teacher_concatenated.summary() """


    second_teacher=loadModelH5(directory+"second_teacher_w_rel_kd")

    teacher_concatenated = tf.keras.Model(inputs=second_teacher.input,outputs=second_teacher.get_layer(layer_until_name_teacher).output)
    teacher_concatenated._name = "second_teacher_intermediate"
    teacher_concatenated.summary()

    student_intermediate = tf.keras.Model(inputs=student.input,outputs=student.get_layer(layer_until_name_student).output)
    student_intermediate._name = "student_intermediate"
    student_intermediate.summary()
    

    """ 
    for i, layer in enumerate(teacher0.layers):
        if i <= 3:  
            print("Layer of teacher is ",layer.name)
            print("Layer of teacher intermd is ",teacher_0_intermediate.layers[i-1].name)
            
            teacher_0_intermediate.layers[i-1].set_weights(layer.get_weights())

    teacher_0_intermediate.trainable = False
    teacher_0_intermediate.summary() """
    

    print("Teachers loaded")

class Distiller_fitnet(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_fitnet, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """  self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])

        self.student_aux_layers_v3._name="student_aux_v3" """
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

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            #aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

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
                tf.nn.softmax(teacher_prediction, axis=1),
                tf.nn.softmax(student_prediction, axis=1)
            )
            distillation_loss = distillation_loss*50
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

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, student_in,layer_until_in):
        super(CustomCallback, self).__init__()
        self.student = student_in
        self.layer_until=layer_until_in
    def on_epoch_end(self, epoch, logs=None):
        student = self.student
        student_intermediate = tf.keras.Model(inputs=student.input,outputs=student.get_layer(self.layer_until).output)
        student_intermediate._name = "student_intermediate"
        #student_intermediate.summary()

if train_student_with_fitnet_second_t:
    print("TRAINING STUDENT WITH FITNET KD DEEPER ON SECOND") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_second_t_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        layer_until_teacher='Conv-4'

        define_intermediate_networks_v2(layer_until_name_student=layer_until,layer_until_name_teacher=layer_until_teacher,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #Second stage of training

        alpha_rel_kd = trial.suggest_float('alpha_kd', 0, 0.01)
        #alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_fitnet_second_t_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_fitnet_second_t_kd'],
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
            (df['Network'] == 'student_fitnet_second_t_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_fitnet_second_t_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_fitnet_second_t_kd")

        acc_stud = metrics_stud[1]
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234) for AI4EDGE4 from SECOND TEACHER",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        


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

#----------- COMPLETE OVERHAUL OF FEATURE DISTILLATION TRAINING WITH SECOND TEACHER-----------------

class Distiller_cofd(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_cofd, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """ self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_0_intermediate.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])
        self.student_aux_layers_v3._name="student_aux_v3" """
    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha,
            temperature,
    ):
        """ Configure the Distiller_cofd.
    student_loss_fn: Loss function of difference between student
                predictions and ground-truth
    distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
    alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
    temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller_cofd, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def get_margin_v2(self,features):

        std_dev = tf.keras.backend.std(features)
        mean = tf.keras.backend.mean(features)
        # margin = np.where(norm.cdf(-mean / std_dev) > 1e-3,
        #                    - std_dev * np.exp(- (mean / std_dev) ** 2 / 2) / np.sqrt(2 * np.pi) / norm.cdf(-mean / std_dev) + mean,-3 * std_dev).astype(np.float32)
        #margin = std_dev + mean
        eps = tf.constant(1e-3, dtype=tf.float32)

        condition = tf.greater(-mean / std_dev, eps)

        tmp1 = - std_dev * tf.exp(- tf.pow(mean / std_dev, 2) / 2) / tf.sqrt(2 * math.pi)
        tmp2 = tf.math.erf(-mean / std_dev)
        tmp3 = -3 * std_dev

        margin = tf.where(condition, tmp1 / tmp2 + mean, tmp3)
        margin_value =  tf.dtypes.cast(margin, tf.float32)
        return margin_value

    def train_step(self, data):

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            #aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

            teach_feat_augmented = aux(teacher_features, training=False)

            margins_teach = self.get_margin_v2(teach_feat_augmented)

            teach_feat_augmented = tf.stop_gradient(tf.maximum(teach_feat_augmented, margins_teach))

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
            print("Loss in Distiller_cofd :", loss)
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

if train_student_with_cofd_second_t:
    print("TRAINING STUDENT WITH COFD KD") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_cofd_second_teac_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd = trial.suggest_float('alpha_rel_kd', 0.4, 0.6)


        layer_until='Conv-3'
        layer_until_teacher='Conv-4'

        define_intermediate_networks_v2(layer_until_name_student=layer_until,layer_until_name_teacher=layer_until_teacher,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_cofd(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_cofd_second_teac_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_cofd_second_teac_kd'],
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
            (df['Network'] == 'student_cofd_second_teac_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_cofd_second_teac_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_cofd_second_teac_kd")
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        acc_stud = metrics_stud[1]
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in cofd kd (seed 1234) for AI4EDGE4 FOR DEEPER LAYERS FOR SECOND TEACHER",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_cofd_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.55

        #Test
        layer_until='Conv-2'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_cofd(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
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


#---------------FITNET KD WITH SECOND TEACHER - DEEPER -----------------------

class Distiller_fitnet(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_fitnet, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])

        self.student_aux_layers_v3._name="student_aux_v3"
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

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            aux_pool=self.student_aux_layers_v3
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
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


if train_student_with_fitnet_second_t_deeper:
    print("TRAINING STUDENT WITH FITNET KD DEEPER ON SECOND") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_second_t_deeper_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        layer_until_teacher='Conv-5'

        define_intermediate_networks_v2(layer_until_name_student=layer_until,layer_until_name_teacher=layer_until_teacher,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #Second stage of training

        alpha_rel_kd = trial.suggest_float('alpha_kd', 0, 0.01)
        #alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_fitnet_second_t_deeper_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_fitnet_second_t_deeper_kd'],
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
            (df['Network'] == 'student_fitnet_second_t_deeper_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_fitnet_second_t_deeper_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_fitnet_second_t_deeper_kd")

        acc_stud = metrics_stud[1]
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234) for AI4EDGE4 from SECOND TEACHER DEEPER",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        


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

#---------------FITNET KD WITH SECOND TEACHER - EARLIER -----------------------


class Distiller_fitnet(tf.keras.Model):
    def __init__(self, student, teacher,alpha):
        super(Distiller_fitnet, self).__init__()

        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.l = [1e2, 2e2]

        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.he_normal(),
                                                  use_biases = False, activation_fn = None, trainable = True))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = False))
       
        self.student_aux_layers_v2=tf.keras.Sequential([
            tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
            tf.keras.layers.BatchNormalization(trainable=False)])
        self.student_aux_layers_v2._name="student_aux_v2"

        """ self.student_aux_layers_v3=tf.keras.Sequential([
        tcl.Conv2d([1, 1], teacher_concatenated.output_shape[-1]),
        tf.keras.layers.BatchNormalization(trainable=False),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')])

        self.student_aux_layers_v3._name="student_aux_v3" """
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

        global teacher_features
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        # print("Teacher prediction   ...", teacher_prediction)
        
        #teacher_features = student_intermediate(x, training=False)

        teacher_features=teacher_concatenated(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_prediction = self.student(x, training=True)

            student_features = student_intermediate(x, training=False)
            # # Compute losses
            student_loss = self.student_loss_fn(y, student_prediction)
           


            aux = self.student_aux_layers_v2

            """ aux_pool=self.student_aux_layers_v3 """
            # Define the new model with a MaxPooling2D layer
            
            #stud_feat_augmented = aux_pool(student_features, training=False)
            stud_feat_augmented = aux(student_features, training=False)

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


if train_student_with_fitnet_second_t_earlier:
    print("TRAINING STUDENT WITH FITNET KD EARLIER ON SECOND") 
    def objective(trial):
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_second_t_earlier_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        layer_until_teacher='Conv-3'

        define_intermediate_networks_v2(layer_until_name_student=layer_until,layer_until_name_teacher=layer_until_teacher,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)
        
        #Second stage of training

        alpha_rel_kd = trial.suggest_float('alpha_kd', 0, 0.01)
        #alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=second_teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        student_fitnet = distiller.student

        print("Quantizar o dataset started")
        # test
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        fold_model_filename="student_fitnet_second_t_earlier_kd.tflite"
        path_student = testNameDir + "/" + fold_model_filename
        # if teacher model isnt quantized
        #if not os.path.isfile(path_student):
        
        metrics_stud = student_fitnet.evaluate(X_validation,Y_validation)

        data = {
                'Network': ['student_fitnet_second_t_earlier_kd'],
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
            (df['Network'] == 'student_fitnet_second_t_earlier_kd') & (
                    df['Testing Accuracy'] > metrics_stud[1])]
        
        if filtered_df.empty:
            print("Convertendo student")
            converter = tf.lite.TFLiteConverter.from_keras_model(student_fitnet)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
            converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()

            with open(directory+"student_fitnet_second_t_earlier_kd.tflite", "wb") as output_file:
                output_file.write(tflite_model)
            
            student_fitnet.save(directory+"student_fitnet_second_t_earlier_kd")

        acc_stud = metrics_stud[1]
        del student_fitnet
        del student_scratch
        del distiller
        del callback_list
        del history
        return acc_stud    
    if train_optuna:
        study = optuna.create_study(study_name="Changing alpha in fitnets-based kd (seed 1234) for AI4EDGE4 from SECOND TEACHER EARLIER",
                                    direction='maximize',
                                    storage="sqlite:///optuna_results.db", load_if_exists=True)
        study.optimize(objective, n_trials=optuna_trials)
    else:
        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        student_fitnet = BM2(input_Shape)
        student_fitnet._name = "student_fitnet_kd"

        # Clone student for later comparison
        student_scratch = tf.keras.models.clone_model(student_fitnet)
        alpha_rel_kd=0.001

        #Test
        layer_until='Conv-3'
        define_intermediate_networks(layer_until_name=layer_until,student=student_fitnet)

        # Initialize and compile distiller
        #distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)
        distiller = Distiller_fitnet(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            CustomCallback(student_in=distiller.student,layer_until_in=layer_until)
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        alpha_rel_kd=0.001
        distiller = Distiller_response(student=student_fitnet, teacher=teacher,alpha=alpha_rel_kd)

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
            tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=0, mode='max', restore_best_weights=True),
            #CustomCallback()
        ]
        # Distill teacher to student
        
        history = distiller.fit(X_train, Y_train, epochs=int((epocas/2)), batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

        


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

#train_student_with_fitnet_second_t_earlier=True

#---------------TRAIN STUDENT WO KD -----------------------

if train_std_wo_kd:
    input_Shape = (X_train.shape[1], X_train.shape[2], 1)
    student_wo_kd = BM2(input_Shape)
    student_wo_kd._name = "Student"
    # Clone student for later comparison
    student_scratch = tf.keras.models.clone_model(student_wo_kd)

    # Initialize and compile distiller
    distiller = Distiller(student=student_wo_kd, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),  # keras.optimizers.Adam(),
        metrics=['categorical_accuracy'],  # [keras.metrics.SparseCategoricalAccuracy()], #['categorical_accuracy'],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        # 'categorical_crossentropy',#keras.losses.SparseCategoricalCrossentropy(from_logits=True),#tf.keras.losses.CategoricalCrossentropy(),categorical_crossentropy
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=1,
        temperature=3,
    )

    callback_list = [
        dcase_util.tfkeras.StasherCallback(
            epochs=epocas, initial_delay=10, monitor='val_categorical_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    ]
    #student.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['categorical_accuracy'])
    # Distill teacher to student
    history = distiller.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,validation_data=(X_validation, Y_validation), shuffle=True, verbose=1,callbacks=callback_list)

    student_wo_kd.save(directory+"student_wo_kd")

    student_wo_kd.evaluate(X_validation,Y_validation)


print("----------------TESTING-------------------")

""" print("Testing teacher")
teacher.summary()
teacher.evaluate(X_validation,Y_validation)


print("Testing initial student")
student.summary()

student.evaluate(X_validation,Y_validation) """
""" 
if os.path.isdir(directory+"student_rel_kd"):
    print("Testing relational-based student")
    student_rel_kd=loadModelH5(directory+"student_rel_kd")
    student_rel_kd.evaluate(X_validation,Y_validation)

if os.path.isdir(directory+"student_rel_resp_kd"):
    print("Testing relational and response based student")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_kd")
    student_rel_kd.evaluate(X_validation,Y_validation)

if os.path.isdir(directory+"student_fitnet_kd"):

    print("Testing Fitnet based student")
    student_rel_kd=loadModelH5(directory+"student_fitnet_kd")
    student_rel_kd.evaluate(X_validation,Y_validation)
 """
if os.path.isdir(directory+"student_rel_resp_v2_kd"):

    print("Testing relational and response based student with stagewise")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_v2_kd")
    student_rel_kd.evaluate(X_validation,Y_validation)

""" if os.path.isdir(directory+"second_teacher_rel_resp_kd"):

    print("Testing teacher_rel_resp_kd")
    teacher_rel=loadModelH5(directory+"second_teacher_rel_resp_kd")
    teacher_rel.evaluate(X_validation,Y_validation)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(teacher_rel)
    tflite_model = converter.convert()

    with open(directory+"second_teacher_rel_resp_kd.tflite", "wb") as output_file:
                    output_file.write(tflite_model)
 """
def format_name(name):
    # Check if the name contains any '/' characters
    if '/' in name:
        # Split the name on the '/' character
        parts = name.split('/')
        # Return the second part of the split name
        return parts[1]
    else:
        # If the name does not contain any '/' characters, return it unchanged
        return name

if test_rrs_qat:

    print("Testing relational and response based student with stagewise")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_v2_kd")

    converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
    tflite_model = converter.convert()
    #saving converted model in "converted_model.tflite" file
    open("std_rrs__not_quantized.tflite", "wb").write(tflite_model)

    student_rel_kd=loadModelH5(directory+"student_rel_resp_v2_kd")
    converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    #saving converted model in "converted_quant_model.tflite" file
    open("std_rrs_quantized.tflite", "wb").write(tflite_quant_model)

    random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.002),replace=False)  # seleciona random elements X_train
    representative_data = X_train[random_indices, :]
    print("Full size of train dataset:")
    print(round((X_train.shape[0])))

    x_test_normalized = representative_data
    def representative_dataset():
        for x in x_test_normalized:
            yield [np.array([x], dtype=np.float32)]

    student_rel_kd=loadModelH5(directory+"student_rel_resp_v2_kd")
    converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()
    #saving converted model in "converted_quant_model.tflite" file
    open("std_rrs_quantized_fullint8.tflite", "wb").write(tflite_quant_model)


    qat_model_path=directory+"student_rrs_kd_qat.tflite"

    #import os
    print("Float model in Mb:", os.path.getsize('std_rrs__not_quantized.tflite') / float(2**20))
    print("Quantized model in Mb:", os.path.getsize('std_rrs_quantized.tflite') / float(2**20))
    print("Compression ratio:", os.path.getsize('std_rrs__not_quantized.tflite')/os.path.getsize('std_rrs_quantized.tflite'))
    print("QAT model in Mb:", os.path.getsize(qat_model_path) / float(2**20))
    print("Compression ratio:", os.path.getsize('std_rrs__not_quantized.tflite')/os.path.getsize(qat_model_path))
    print("Quantized full int8 model in Mb:", os.path.getsize('std_rrs_quantized_fullint8.tflite') / float(2**20))
    print("Compression ratio:", os.path.getsize('std_rrs__not_quantized.tflite')/os.path.getsize('std_rrs_quantized_fullint8.tflite'))

    
   
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="std_rrs_quantized.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on some input data.
    input_shape = input_details[0]['shape']
    acc=0
    for i in range(len(X_validation)):
        input_data = X_validation[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if(np.argmax(output_data) == np.argmax(Y_validation[i])):
            acc+=1
    acc = acc/len(X_validation)
    print("Quantized model, accuracy:")
    print(acc*100)

     # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="std_rrs_quantized_fullint8.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on some input data.
    input_shape = input_details[0]['shape']
    acc=0
    for i in range(len(X_validation)):
        input_data = X_validation[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if(np.argmax(output_data) == np.argmax(Y_validation[i])):
            acc+=1
    acc = acc/len(X_validation)
    print("Quantized model FULL INT8, accuracy:")
    print(acc*100)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="std_rrs__not_quantized.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on some input data.
    input_shape = input_details[0]['shape']
    acc=0
    for i in range(len(X_validation)):
        input_data = X_validation[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if(np.argmax(output_data) == np.argmax(Y_validation[i])):
            acc+=1
    acc = acc/len(X_validation)
    print("NOT quantized model, accuracy:")
    print(acc*100)

    # Compute the scaling factor
    scale = (np.max(X_validation) - np.min(X_validation)) / 255.0

    # Quantize the input data
    X_validation_quantized = (X_validation / scale).astype(np.int8)

    interpreter = tf.lite.Interpreter(model_path=qat_model_path)
    #directory+"student_rrs_kd_qat.tflite"
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on some input data.
    input_shape = input_details[0]['shape']
    acc=0
    for i in range(len(X_validation_quantized)):
        input_data = X_validation_quantized[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if(np.argmax(output_data) == np.argmax(Y_validation[i])):
            acc+=1
    acc = acc/len(X_validation_quantized)
    print("QAT model, accuracy:")
    print(acc*100)

    # test



    """ print("Testing relational and response based student with stagewise")
    student_rel_kd=loadModelH5(directory+"student_rel_resp_v2_kd")
    converter = tf.lite.TFLiteConverter.from_keras_model(student_rel_kd)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int8  #tf23 da erro com esta linha
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
    converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()


    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    #interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print("Expected input shape of the model:", input_shape)

    # Print the shape of X_validation
    print("Shape of X_validation:", X_validation.shape)

        # Convert X_validation to INT8
    X_validation_int8 = X_validation.astype(np.int8)

    predictions = []
    for x_val in X_validation_int8:
        # Preprocess input if necessary
        # ...
        x_val = np.expand_dims(x_val, axis=0)

        #print(input_details[0]['index'])
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], x_val)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Postprocess output if necessary
        # ...

        # Collect predictions
        predictions.append(output)


    # Convert predictions to a numpy array
    predictions = np.array(predictions)

    # Remove extra dimensions from predictions
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.astype(np.int8)

    # Convert Y_validation to INT8
    Y_validation = np.squeeze(Y_validation)
    # Convert Y_validation_int8 to a 1D array of labels
    Y_validation = np.argmax(Y_validation, axis=1)
    Y_validation_int8 = Y_validation.astype(np.int8)

        # Ensure Y_validation has the appropriate shape
    #Y_validation_int8 = Y_validation_int8.squeeze()
    print("Shape of Y_validation_int8:", Y_validation_int8.shape)
    print("Data type of Y_validation_int8:", Y_validation_int8.dtype)
    print("Shape of predictions:", predictions.shape)
    print("Data type of predictions:", predictions.dtype)
    # Compare predictions with ground truth
    # Convert predictions to a 1D array of predicted labels

    accuracy = accuracy_score(Y_validation_int8, predictions)
    print("Accuracy is:")
    print(accuracy) """
    # Convert TensorFlow model to Keras model
    #keras_model = tf.keras.models.model_from_tf(loaded_model)
    #keras_model.evaluate(X_validation,Y_validation)