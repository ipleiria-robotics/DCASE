import h5py
from myLiB.funcoesPipline import *
from myLiB.plots import *
from myLiB.utils import *

matplotlib.use('Agg')
# matplotlib.use('TKAgg')
from NeSsi import nessi
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

# Baseline dataset
dasetName="BaselineData"
hdf5_path_Train = "data/0.Baseline/Train_fs44100_40_2048_0.04_0.02.h5"
hdf5_path_Test = "data/0.Baseline/Test_fs44100_40_2048_0.04_0.02.h5"

# # kerasTuner8k_260_8
# dasetName="kerasTuner8k_260_8"
# hdf5_path_Train = "data/8k_260_8/Train_fs8000_260_2048_0.256_0.128.h5"
# hdf5_path_Test = "data/8k_260_8/Test_fs8000_260_2048_0.256_0.128.h5"

# # kerasTuner8k_run2
# dasetName="kerasTuner8k_run2"
# hdf5_path_Train = "data/8k_140_8/Train_fs8000_140_2048_0.256_0.128.h5"
# hdf5_path_Test = "data/8k_140_8/Test_fs8000_140_2048_0.256_0.128.h5"
# dasetName="kerasTuner8k_run2_SpecAug"
# hdf5_path_Train = "data/8k_140_8/SpecAug/Train_fs8000_140_2048_0.256_0.128.h5"
# hdf5_path_Test = "data/8k_140_8/SpecAug/Test_fs8000_140_2048_0.256_0.128.h5"
# dasetName="kerasTuner8k_run2_wavAug"
# hdf5_path_Train = "data/8k_140_8/wavAug/Train_fs8000_140_2048_0.256_0.128.h5"
# hdf5_path_Test = "data/8k_140_8/wavAug/Test_fs8000_140_2048_0.256_0.128.h5"
# dasetName="kerasTuner8k_run2_WavSpecAug"
# hdf5_path_Train = "data/8k_140_8/WavSpecAug/Train_fs8000_140_2048_0.256_0.128.h5"
# hdf5_path_Test = "data/8k_140_8/WavSpecAug/Test_fs8000_140_2048_0.256_0.128.h5"


numTestes=1
epocas = 200
batch_size = 64
Otimizer_type="Adam" #"Adam" RMSprop,SGD
Lrate=0.001
model_type="Baseline"
testNameDir="test/"+model_type+"/0."+dasetName+"_E"+str(epocas)+"_B"+str(batch_size)+"_"+Otimizer_type

if os.path.isdir(testNameDir): pass
else: os.makedirs(testNameDir)

dcase_util.utils.setup_logging(logging_file=os.path.join(testNameDir+"/"+"task1a_v2_em.log"))
log = dcase_util.ui.ui.FancyLogger()
log.title('DCASE2022 / Task1A -- low-complexity Acoustic Scene Classification')
log.line()


arraytest=np.arange(numTestes)
for i in arraytest:
    timer.start()
    log.section_header('learning')
    fold_model_filename = "model_"+str(i)+".tflite"
    if not os.path.isfile(testNameDir+"/"+fold_model_filename):
        with h5py.File(hdf5_path_Train, 'r') as hf:
            print(hf.keys())
            X_train = np.array(hf['X_train'])
            X_validation = np.array(hf['X_validation'])
            Y_validation = [x.decode() for x in hf['Y_validation']]
            Y_train = [x.decode() for x in hf['Y_train']]
            descricao = [x.decode() for x in hf['descricao']]
            descricao = "".join([str(elem) for elem in descricao])
        log.line()
        log.line("DATA INFO")
        log.line(descricao)
        log.line()
        log.line("MODEL INFO")
        log.line(model_type)
        log.line()

        Y_train = labelsEncoding('Train', scene_labels, Y_train)  #'train'= 'smooth_labels'|'Mysmooth_labels'|'Mysmooth_labels2'
        Y_validation = labelsEncoding('Val', scene_labels, Y_validation)

        do_learning(X_train,Y_train,X_validation,Y_validation,fold_model_filename,log,testNameDir,model_type,Otimizer_type,Lrate,epocas,batch_size)
    timer.stop()
    log.foot(time=timer.elapsed())

    macc, params = nessi.get_model_size(testNameDir + "/" + fold_model_filename, 'tflite')
    nessi.validate(macc, params, log)

    log.section_header('testing')
    timer.start()
    path_estimated_scene = "res_fold_"+str(i)+".csv"
    if not os.path.isfile(testNameDir+"/"+path_estimated_scene):
        # Loop over all cross-validation folds and learn acoustic models
        with h5py.File(hdf5_path_Test, 'r') as hf:
            print(hf.keys())
            # features = int16_to_float32(hf['features'][index_files])
            test_features = np.array(hf['features'])
            test_filename = [x.decode() for x in hf['filename']]
            # scene_label = [x.decode() for x in hf['scene_label']]

        # test_features = librosa.feature.delta(test_features, order=1)

        do_testing(scene_labels, fold_model_filename, path_estimated_scene, test_features, test_filename, log,testNameDir)
    timer.stop()
    log.foot(time=timer.elapsed())

    log.section_header('evaluation')
    timer.start()
    # do_evaluation(path_estimated_scene,log)
    globals()[f"df{i}"] = do_evaluation(path_estimated_scene,log,testNameDir,fold_model_filename)
    timer.stop()
    log.foot(time=timer.elapsed())

if(numTestes>1):
    log.section_header('Media dos testes realizados')
    dMean = pd.concat([globals()[f"df{i}"] for i in arraytest])
    dMean=dMean.groupby(level=0).mean()
    dMean.insert(0,"Scene", df1["Scene"])
    log.line(dMean.to_string())

