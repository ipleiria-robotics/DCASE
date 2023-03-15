
import dcase_util

from myLiB.funcoesPipline import *
from myLiB.models import *
from myLiB.optimizers import *
from myLiB.plots import *
from myLiB.utils import *
import os
import sed_eval

matplotlib.use('Agg')
# matplotlib.use('TKAgg')

def do_learning(X_train,Y_train,X_validation,Y_validation,fold_model_filename,log,testNameDir,model_type,Otimizer_type,Lrate,epocas,batch_size):
    """Learning stage

    Parameters
    ----------

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing
    """
    X_train = np.expand_dims(X_train, -1)
    Y_train = np.expand_dims(Y_train, -1)
    X_validation = np.expand_dims(X_validation, -1)
    Y_validation = np.expand_dims(Y_validation, -1)

    # Loop over all cross-validation folds and learn acoustic models
    if not os.path.isfile(testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5")):

        input_Shape=(X_train.shape[1], X_train.shape[2], 1)
        keras_model=modelSelector(model_type, input_Shape)

        optimizer=otimizerSelector(Otimizer_type,Lrate) #Otimizer_type,Lrate

        keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy']) #loss=tfr.keras.losses.SoftmaxLoss()
        keras_model.summary(print_fn=log.info)

        #Create callback list
        callback_list = [
            dcase_util.tfkeras.ProgressLoggerCallback(
                epochs=epocas,
                metric='categorical_accuracy',
                loss='categorical_crossentropy',
                output_type='logging'
            ),
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=20, monitor='val_categorical_accuracy'
            )

        ]

        history = keras_model.fit(X_train, Y_train,epochs=epocas, batch_size=batch_size, validation_data=(X_validation, Y_validation),shuffle=True, verbose=0, callbacks=callback_list)

        for callback in callback_list:
            if isinstance(callback, dcase_util.tfkeras.StasherCallback):
                # Fetch the best performing model
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    keras_model.set_weights(best_weights)
                break

        plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                     history.history['loss'],
                     history.history['val_loss'], png_name=testNameDir+"/"+"Training_acc_loss_"+fold_model_filename+".png")

        try:
            #keras_model.save(testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5"))
            saveModelH5(keras_model, testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5"))
        except:
            log.line("erro save model h5")



        # keras_model=loadModelH5(testNameDir+"/"+fold_model_filename.replace(".tflite", ".h5"))

        # Quantization to int8
        # A generator that provides a representative dataset
        # class BatchGenerator():
        #     def __init__(self,
        #                  hdf5_path,
        #                  batch_size=32):
        #         self.hdf5_path = hdf5_path
        #         self.batch_size = batch_size
        #
        #     def __call__(self):
        #         index_in_hdf5 = np.arange(self.batch_size)
        #         with h5py.File(self.hdf5_path, 'r') as hf:
        #             #features = int16_to_float32(hf['features'][index_in_hdf5])
        #             # features = hf['features'][index_in_hdf5]
        #             features = hf['X_train'][index_in_hdf5]
        #         for feat in features:
        #             yield ([tf.expand_dims(tf.expand_dims(feat, -1), 0)])  #np.expand_dims(feat, -1)
        #
        # batch_generator = BatchGenerator(hdf5_path, batch_size=100)

        # # Quantization to int8
        # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT] # converts to int32
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8  #BASELINE ERA ASSIM-->[tf.int8] EM TF 2.3 DA ERRO
        # converter.inference_output_type = tf.int8  # or tf.uint8 #BASELINE ERA ASSIM-->[tf.int8] EM TF 2.3 DA ERRO
        # converter.representative_dataset = batch_generator
        # tflite_model = converter.convert()
        #
        # # Save the quantized model
        # with open('model2.tflite', "wb") as output_file:
        #     output_file.write(tflite_model)

        #Save GRU model
        # keras_model.save_weights(testNameDir+"/"+"crnn.h5")
        # modelo = crnn11(input_Shape)
        # modelo.load_weights(testNameDir+"/"+"crnn.h5")
    else:
        keras_model = loadModelH5(testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5"))

    if(keras_model):
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data
        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        converter.representative_dataset = representative_dataset
        # converter.experimental_new_converter = True
        # converter._experimental_new_quantizer = False
        tflite_model = converter.convert()
        with open(testNameDir + "/" + fold_model_filename, "wb") as output_file:
            output_file.write(tflite_model)

        # x_test_normalized = representative_data
        # def representative_dataset():
        #     for x in x_test_normalized:
        #         yield [np.array([x], dtype=np.float32)]
        #
        # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        # converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        # converter.representative_dataset = representative_dataset
        # converter.experimental_new_converter = True
        # # converter._experimental_new_quantizer = False
        # tflite_model = converter.convert()
        # with open(testNameDir + "/" + fold_model_filename, "wb") as output_file:
        #     output_file.write(tflite_model)

def do_learning2(X_train, Y_train, X_validation, Y_validation, fold_model_filename, log, testNameDir, model_type,Otimizer_type, Lrate, epocas, batch_size):
    """Learning stage

    Parameters
    ----------

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing
    """
    X_train = np.expand_dims(X_train, -1)
    Y_train = np.expand_dims(Y_train, -1)
    X_validation = np.expand_dims(X_validation, -1)
    Y_validation = np.expand_dims(Y_validation, -1)

    # Loop over all cross-validation folds and learn acoustic models
    if not os.path.isfile(testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5")):

        input_Shape = (X_train.shape[1], X_train.shape[2], 1)
        keras_model = modelSelector(model_type, input_Shape)

        optimizer = otimizerSelector(Otimizer_type, Lrate)  # Otimizer_type,Lrate

        keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])  # loss=tfr.keras.losses.SoftmaxLoss()
        keras_model.summary(print_fn=log.info)

        # Create callback list
        callback_list = [
            dcase_util.tfkeras.ProgressLoggerCallback(
                epochs=epocas,
                metric='categorical_accuracy',
                loss='categorical_crossentropy',
                output_type='logging'
            ),
            dcase_util.tfkeras.StasherCallback(
                epochs=epocas, initial_delay=20, monitor='val_categorical_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, verbose=0, mode='min')#,restore_best_weights=True)
        ]

        history = keras_model.fit(X_train, Y_train, epochs=epocas, batch_size=batch_size,
                                  validation_data=(X_validation, Y_validation), shuffle=True, verbose=0,
                                  callbacks=callback_list)

        for callback in callback_list:
            if isinstance(callback, dcase_util.tfkeras.StasherCallback):
                # Fetch the best performing model
                callback.log()
                best_weights = callback.get_best()['weights']
                if best_weights:
                    keras_model.set_weights(best_weights)
                break

        plot_History(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'],
                     history.history['loss'],
                     history.history['val_loss'],
                     png_name=testNameDir + "/" + "Training_acc_loss_" + fold_model_filename + ".png")

        try:
            # keras_model.save(testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5"))
            saveModelH5(keras_model, testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5"))
        except:
            log.line("erro save model h5")

        # keras_model=loadModelH5(testNameDir+"/"+fold_model_filename.replace(".tflite", ".h5"))

        # Quantization to int8
        # A generator that provides a representative dataset
        # class BatchGenerator():
        #     def __init__(self,
        #                  hdf5_path,
        #                  batch_size=32):
        #         self.hdf5_path = hdf5_path
        #         self.batch_size = batch_size
        #
        #     def __call__(self):
        #         index_in_hdf5 = np.arange(self.batch_size)
        #         with h5py.File(self.hdf5_path, 'r') as hf:
        #             #features = int16_to_float32(hf['features'][index_in_hdf5])
        #             # features = hf['features'][index_in_hdf5]
        #             features = hf['X_train'][index_in_hdf5]
        #         for feat in features:
        #             yield ([tf.expand_dims(tf.expand_dims(feat, -1), 0)])  #np.expand_dims(feat, -1)
        #
        # batch_generator = BatchGenerator(hdf5_path, batch_size=100)

        # # Quantization to int8
        # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT] # converts to int32
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8  #BASELINE ERA ASSIM-->[tf.int8] EM TF 2.3 DA ERRO
        # converter.inference_output_type = tf.int8  # or tf.uint8 #BASELINE ERA ASSIM-->[tf.int8] EM TF 2.3 DA ERRO
        # converter.representative_dataset = batch_generator
        # tflite_model = converter.convert()
        #
        # # Save the quantized model
        # with open('model2.tflite', "wb") as output_file:
        #     output_file.write(tflite_model)

        # Save GRU model
        # keras_model.save_weights(testNameDir+"/"+"crnn.h5")
        # modelo = crnn11(input_Shape)
        # modelo.load_weights(testNameDir+"/"+"crnn.h5")
    else:
        keras_model = loadModelH5(testNameDir + "/" + fold_model_filename.replace(".tflite", ".h5"))

    if (keras_model):
        # seleciona 50% dos dados para quantizar em INT8
        random_indices = np.random.choice(X_train.shape[0], size=round((X_train.shape[0]) * 0.5),replace=False)  # seleciona random elements X_train
        # representative_data = X_train.numpy()[random_indices, :] # Aumentar a percentagem de elementos na quantização antes do envio final melhores resultados
        representative_data = X_train[random_indices, :]

        x_test_normalized = representative_data

        def representative_dataset():
            for x in x_test_normalized:
                yield [np.array([x], dtype=np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        converter.representative_dataset = representative_dataset
        # converter.experimental_new_converter = True
        # converter._experimental_new_quantizer = False
        tflite_model = converter.convert()
        with open(testNameDir + "/" + fold_model_filename, "wb") as output_file:
            output_file.write(tflite_model)

        # x_test_normalized = representative_data
        # def representative_dataset():
        #     for x in x_test_normalized:
        #         yield [np.array([x], dtype=np.float32)]
        #
        # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # converts to int32  #tf23 da erro com esta linha
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8  #[tf.int8]
        # converter.inference_output_type = tf.int8  # or tf.uint8 #[tf.int8]
        # converter.representative_dataset = representative_dataset
        # converter.experimental_new_converter = True
        # # converter._experimental_new_quantizer = False
        # tflite_model = converter.convert()
        # with open(testNameDir + "/" + fold_model_filename, "wb") as output_file:
        #     output_file.write(tflite_model)

def do_testing(scene_labels,fold_model_filename,path_estimated_scene,test_features,test_filename, log,testNameDir):
    """Testing
    Parameters
    ----------

    scene_labels : list of str
        List of scene labels

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    """
    # Loop over all cross-validation folds and test

    # Load the model into an interpreter
    if(tf.__version__=='2.1.0'):
        interpreter = tf.lite.Interpreter(model_path=testNameDir+"/"+fold_model_filename)
    if(tf.__version__>='2.3.0'):
        interpreter = tf.lite.Interpreter(model_path=testNameDir+"/"+fold_model_filename,num_threads=14) #TF2.3
    interpreter.allocate_tensors()

    if not os.path.isfile(testNameDir+"/"+path_estimated_scene):

        # Initialize results container
        res = dcase_util.containers.MetaDataContainer(
            filename=testNameDir+"/"+path_estimated_scene
        )

        # Get input and output indexes for the interpreter
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        try:
            # TODO: comentei e fiz alterações
            for i, feat in enumerate(test_features):
                # Get feature filename
                filename = test_filename[i]

                if input_details['dtype'] == np.int8 or input_details['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details["quantization"]
                    feat = feat / input_scale + input_zero_point
                #     # feat = np.expand_dims(feat, 0).astype(input_details["dtype"]) #np array
                #     feat = tf.expand_dims(feat, 0) #tensor
                #     feat = tf.cast(feat, dtype=input_details["dtype"], name=None)  # Tensor
                # else:
                #     # feat = np.expand_dims(feat, 0).astype(input_details["dtype"])
                #     feat = tf.expand_dims(feat, 0) #tensor
                #     feat = tf.cast(feat, dtype=input_details["dtype"], name=None)


                #TESTAR TEMPOS COM np array ou com tensor tf
                # feat = np.expand_dims(feat, 0).astype(input_details["dtype"])
                feat = tf.expand_dims(feat, 0)  # tensor
                feat = tf.cast(feat, dtype=input_details["dtype"], name=None)

                input_data = tf.expand_dims(feat, -1)  #np.expand_dims(X_train, -1)

                # Get network output
                interpreter.set_tensor(input_details['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])

                if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
                    output_scale, output_zero_point = output_details['quantization']
                    output = output_scale * (output.astype(np.float32) - output_zero_point)

                probabilities = output.T
                # Clean up internal states.
                interpreter.reset_all_variables()

                #necessario
                probabilities = dcase_util.data.ProbabilityEncoder().collapse_probabilities(
                    probabilities=probabilities,
                    operator='sum',
                    time_axis=1
                )

                # Binarization of the network output
                frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                    probabilities=probabilities,
                    binarization_type='frame_max',
                    threshold= 0.5
                )

                estimated_scene_label = dcase_util.data.DecisionEncoder(
                    label_list=scene_labels
                ).majority_vote(
                    frame_decisions=frame_decisions
                )

                # Collect class wise probabilities and scale them between [0-1]
                class_probabilities = {}
                for scene_id, scene_label in enumerate(scene_labels):
                    class_probabilities[scene_label] = probabilities[scene_id] / input_data.shape[0]

                res_data = {
                    'filename': filename,
                    'scene_label': estimated_scene_label
                }
                # Add class class_probabilities
                res_data.update(class_probabilities)

                # Store result into results container
                res.append(
                    res_data
                )
        except Exception as e:
            log.line(e, indent=2)

        if not len(res):
            raise ValueError('No results to save.')

        # Save results container
        fields = ['filename', 'scene_label']
        fields += scene_labels

        res.save(fields=fields, csv_header=True)

def do_testing_h5(scene_labels,fold_model_filename,path_estimated_scene,test_features,test_filename, log):

    model = tf.keras.models.load_model(fold_model_filename)  # same file path

    if not os.path.isfile(path_estimated_scene):

        # Initialize results container
        res = dcase_util.containers.MetaDataContainer(
            filename=path_estimated_scene
        )

        features= test_features
        filenames= test_filename

        try:
            # TODO: comentei e fiz alterações
            for i, feat in enumerate(features):
                # Get feature filename
                filename = filenames[i]

                #feat = np.expand_dims(feat, 0)#.astype(input_details["dtype"])
                feat = feat.reshape(1, feat.shape[0], feat.shape[1], 1).astype(float)  # Alterar o shape se mudar a arquitetura da rede
                output = model.predict(feat, use_multiprocessing=True, workers=10 )

                probabilities = output.T
                # Clean up internal states.

                #necessario
                probabilities = dcase_util.data.ProbabilityEncoder().collapse_probabilities(
                    probabilities=probabilities,
                    operator='sum',
                    time_axis=1
                )

                # Binarization of the network output
                frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                    probabilities=probabilities,
                    binarization_type='frame_max',
                    threshold= 0.5
                )

                estimated_scene_label = dcase_util.data.DecisionEncoder(
                    label_list=scene_labels
                ).majority_vote(
                    frame_decisions=frame_decisions
                )

                # Collect class wise probabilities and scale them between [0-1]
                class_probabilities = {}
                for scene_id, scene_label in enumerate(scene_labels):
                    class_probabilities[scene_label] = probabilities[scene_id] / feat.shape[0]

                res_data = {
                    'filename': filename,
                    'scene_label': estimated_scene_label
                }
                # Add class class_probabilities
                res_data.update(class_probabilities)

                # Store result into results container
                res.append(
                    res_data
                )

        except Exception as e:
            log.line(e, indent=2)

        if not len(res):
            raise ValueError('No results to save.')

        # Save results container
        fields = ['filename', 'scene_label']
        fields += scene_labels

        res.save(fields=fields, csv_header=True)

def do_evaluation(path_estimated_scene,log,testNameDir,fold_model_filename):
    """Evaluation stage

    Parameters
    ----------
    log : dcase_util.ui.FancyLogger
        Logging interface


    Returns
    -------
    nothing

    """
    #Ficheiro base com a informação para avaliar a rede
    path_evaluation = "test/fold1_evaluate.csv"
    all_results = []
    devices = [
        'a',
        'b',
        'c',
        's1',
        's2',
        's3',
        's4',
        's5',
        's6'
    ]
    scene_labels=['airport',
     'bus',
     'metro',
     'metro_station',
     'park',
     'public_square',
     'shopping_mall',
     'street_pedestrian',
     'street_traffic',
     'tram']

    class_wise_results = np.zeros((1 + len(devices), len(scene_labels)))
    class_wise_results_loss = np.zeros((1 + len(devices), len(scene_labels)))

    reference_scene_list = dcase_util.containers.MetaDataContainer().load(
        filename=path_evaluation,
        file_format=dcase_util.utils.FileFormat.CSV,
        csv_header=True,
        delimiter='\t'
    )
    estimated_scene_list = dcase_util.containers.MetaDataContainer().load(
        filename=testNameDir+"/"+path_estimated_scene,
        file_format=dcase_util.utils.FileFormat.CSV,
        csv_header=True,
        delimiter='\t'
    )


    reference_scene_list_devices = {}
    for device in devices:
        reference_scene_list_devices[device] = dcase_util.containers.MetaDataContainer()

    for item_id, item in enumerate(reference_scene_list):
        device = os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[-1]

        reference_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
        reference_scene_list[item_id]['file'] = item.filename

        reference_scene_list[item_id]['source_label']=device

        #reference_scene_list[item_id]['identifier'] = 'identifier': 'barcelona-203-6129',

        reference_scene_list_devices[device].append(item)


    estimated_scene_list_devices = {}
    for device in devices:
        estimated_scene_list_devices[device] = dcase_util.containers.MetaDataContainer()

    for item_id, item in enumerate(estimated_scene_list):
        device = os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[-1]

        estimated_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
        estimated_scene_list[item_id]['file'] = item.filename

        estimated_scene_list[item_id]['source_label'] = device

        estimated_scene_list_devices[device].append(item)

    evaluator = sed_eval.scene.SceneClassificationMetrics(
        scene_labels=scene_labels
    )

    evaluator.evaluate(
        reference_scene_list=reference_scene_list,
        estimated_scene_list=estimated_scene_list
    )

    # Collect data for log loss calculation
    y_true = []
    y_pred = []

    y_true_scene = {}
    y_pred_scene = {}

    y_true_device = {}
    y_pred_device = {}

    estimated_scene_items = {}
    for item in estimated_scene_list:
        estimated_scene_items[item.filename] = item

    scene_labels = scene_labels
    for item in reference_scene_list:
        # Find corresponding item from estimated_scene_list
        estimated_item = estimated_scene_items[item.filename]

        # Get class id
        scene_label_id = scene_labels.index(item.scene_label)
        y_true.append(scene_label_id)

        # Get class-wise probabilities in correct order
        item_probabilities = []
        for scene_label in scene_labels:
            item_probabilities.append(estimated_item[scene_label])

        y_pred.append(item_probabilities)

        if item.scene_label not in y_true_scene:
            y_true_scene[item.scene_label] = []
            y_pred_scene[item.scene_label] = []

        y_true_scene[item.scene_label].append(scene_label_id)
        y_pred_scene[item.scene_label].append(item_probabilities)

        if item.source_label not in y_true_device:
            y_true_device[item.source_label] = []
            y_pred_device[item.source_label] = []

        y_true_device[item.source_label].append(scene_label_id)
        y_pred_device[item.source_label].append(item_probabilities)

    plot_confusion_matrix(np.array(y_true), np.argmax(y_pred, axis=1), scene_labels, normalize=True, title=None, png_name=testNameDir + "/" + "conf_keras" + fold_model_filename + ".png")

    from sklearn.metrics import log_loss
    logloss_overall = log_loss(y_true=y_true, y_pred=y_pred)

    logloss_class_wise = {}
    for scene_label in scene_labels:
        logloss_class_wise[scene_label] = log_loss(
            y_true=y_true_scene[scene_label],
            y_pred=y_pred_scene[scene_label],
            labels=list(range(len(scene_labels)))
        )

    logloss_device_wise = {}
    for device_label in list(y_true_device.keys()):

        logloss_device_wise[device_label] = log_loss(
            y_true=y_true_device[device_label],
            y_pred=y_pred_device[device_label],
            labels=list(range(len(scene_labels)))
        )

    logloss_device_wise = {}
    for device_label in list(y_true_device.keys()):

        logloss_device_wise[device_label] = log_loss(
            y_true=y_true_device[device_label],
            y_pred=y_pred_device[device_label],
            labels=list(range(len(scene_labels)))
        )

    for scene_label_id, scene_label in enumerate(scene_labels):

        class_wise_results_loss[0, scene_label_id] = logloss_class_wise[scene_label]

        for device_id, device_label in enumerate(y_true_device.keys()):
            scene_device_idx = [i for i in range(len(y_true_device[device_label])) if y_true_device[device_label][i] == scene_label_id]
            y_true_device_scene = [y_true_device[device_label][i] for i in scene_device_idx]
            y_pred_device_scene = [y_pred_device[device_label][i] for i in scene_device_idx]
            class_wise_results_loss[1 + device_id, scene_label_id] = log_loss(
                y_true=y_true_device_scene,
                y_pred=y_pred_device_scene,
                labels=list(range(len(scene_labels)))
            )

    results = evaluator.results()
    all_results.append(results)


    evaluator_devices = {}
    for device in devices:
        evaluator_devices[device] = sed_eval.scene.SceneClassificationMetrics(
            scene_labels=scene_labels
        )

        evaluator_devices[device].evaluate(
            reference_scene_list=reference_scene_list_devices[device],
            estimated_scene_list=estimated_scene_list_devices[device]
        )

        results_device = evaluator_devices[device].results()
        all_results.append(results_device)

    for scene_label_id, scene_label in enumerate(scene_labels):
        class_wise_results[0, scene_label_id] = results['class_wise'][scene_label]['accuracy']['accuracy']

        for device_id, device in enumerate(devices):
            class_wise_results[1 + device_id, scene_label_id] = \
                all_results[1 + device_id]['class_wise'][scene_label]['accuracy']['accuracy']

    overall = [
        results['class_wise_average']['accuracy']['accuracy']
    ]
    for device_id, device in enumerate(devices):
        overall.append(all_results[1 + device_id]['class_wise_average']['accuracy']['accuracy'])

    log.line()
    log.row_reset()

    # Table header
    column_headers = ['Scene', 'Logloss']
    column_widths = [16, 10]
    column_types = ['str20', 'float3']
    column_separators = [True, True]
    for dev_id, device in enumerate(devices):
        column_headers.append(device.upper())
        column_widths.append(8)
        column_types.append('float3')
        if dev_id < len(devices) - 1:
            column_separators.append(False)
        else:
            column_separators.append(True)

    column_headers.append('Accuracy')
    column_widths.append(8)
    column_types.append('float1_percentage')
    column_separators.append(False)

    log.row(
        *column_headers,
        widths=column_widths,
        types=column_types,
        separators=column_separators,
        indent=3
    )
    log.row_sep()

    df = pd.DataFrame(columns=column_headers)

    # Class-wise rows
    for scene_label_id, scene_label in enumerate(scene_labels):
        row_data = [scene_label]
        for id in range(class_wise_results_loss.shape[0]):
            row_data.append(class_wise_results_loss[id, scene_label_id])
        row_data.append(class_wise_results[0,scene_label_id]* 100.0)
        log.row(*row_data)

        df_length = len(df)
        df.loc[df_length] = row_data
    log.row_sep()

    # Last row
    column_values = ['Logloss']
    column_values.append(logloss_overall)
    column_types.append('float3')

    for device_label in devices:
        column_values.append(logloss_device_wise[device_label])
        # df = df.append(pd.Series(column_values, index=df.columns[:len(column_values)]), ignore_index=True)

    column_values.append(' ')
    df = df.append(pd.Series(column_values, index=df.columns[:len(column_values)]), ignore_index=True)

    log.row(
        *column_values,
        types=column_types
    )

    column_values = ['Accuracy', ' ']
    column_types = ['str20', 'float1_percentage']
    for device_id, device_label in enumerate(devices[0:]):
        column_values.append(np.mean(class_wise_results[device_id+1,:])*100)
        column_types.append('float1_percentage')

    column_values.append(np.mean(class_wise_results[0, :]) * 100)
    column_types.append('float1_percentage')

    df = df.append(pd.Series(column_values, index=df.columns[:len(column_values)]), ignore_index=True)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    log.row(
        *column_values,
        types=column_types,
    )
    log.line()
    return df


