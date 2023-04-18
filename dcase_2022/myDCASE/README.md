# DCASE2022 - Task 1

> Participação no concurso (DCASE2022) Tarefa 1 Low-Complexity Acoustic Scene Classification
>  [DCASE2022](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification)
>  https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_AI4EDGE_58_t1.pdf

##  Dataset

Change "data" folder with :
https://drive.google.com/drive/folders/1Rpbba1JrqWx45bkhsjSmab9b5EiDlKB6?usp=share_link


## 💻 Requirements
* `conda create --name tf2-dcase python=3.6`
* `conda activate tf2-dcase`
* `conda install ipython`
* `conda install numpy`
* `conda install tensorflow-gpu=2.1.0`
* `conda install -c anaconda cudatoolkit`
* `conda install -c anaconda cudnn`
* `pip install librosa`
* `pip install absl-py==0.9.0`
* `pip install sed_eval`
* `pip install pyyaml==5.3.1`
* `pip install dcase_util`
* `pip install pandas`
* `pip install pyparsing==2.2.0`

[Nessi](https://github.com/AlbertoAncilotto/NeSsi)
* `pip install pathlib`
* `pip install flatbuffers`
* `pip install prettytable`

* `pip install tensorflow-addons`


## 🚀 Lib's, CuDNN, cuda 

```
https://www.tensorflow.org/install/source_windows?hl=pt-br
cuDNN - 7.6 --> https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip?bk9_NEXY_L1Dt3jaUUcxx278vc39i8-MK-hgfenUK8SWO9_GPyJftcm3xAWPu4B2MeG6sTUtqy4KtQuRzDe6doma2fkzLr6YMUCaHVe62E2Te8FoEcQ_5HeRqfiY4uOb6gdPzg3UepDcFAOQF465AGyhwW5UnyGTzs5cATd4Z1WmbYgGpCraLsYnzJ7w5cpN9MjQGpP4oHmAwIHcFcSt5StjECvHj0xjJIdp39M=&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9
cuda - 10.1 --> https://developer.download.nvidia.com/compute/cuda/10.1/secure/Prod/network_installers/cuda_10.1.105_win10_network.exe?MPLH4r3jT3_TF1e-nM773Y3jroGdEnhJ9KtW571osyCuzfSe-amXZOqrIXGYYek8bWKCoTUCDrLVl9tOQzwCDXF1somvRTfAIf93s_8U4OWCFYoTh9wMHjo-PUhnVwmORh0ztoZBlCKjVf5IpOtMYbg9e_mQQWTwaMfFj1CccfP0aFgBpuBbsADG_IPl_YQ=&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9

erro CUDART64_101.DLL anaconda  
https://www.dll-files.com/search/?q=CUDART64_101.DLL
Put it in my conda environment at C:\Users\<user>\Anaconda3\envs\<env name>\Library\bin

https://github.com/iver56/audiomentations
https://github.com/keunwoochoi/kapre
`````

`````
Folder Structure

├──dcase2022 
    ├── dcase2022_task1_baseline-main
         ├── dataset
    ├── myDcase  
        ├── data             
            ├── 0.Baseline (espectogramas .h5 extraidos do dataset config baseline)
            ├── 8k_140_8 (espectogramas .h5 extraidos do dataset config 8k_140_8 (8k, 140mels))          
            ├── 0.datasetInfo.py  (info dataset)
            ├── 1.featuresAug.py (gerar espectogramas .h5 prontos para treino e teste com e sem aumento de dados)
            ├── 2.featuresWave.py (gerar wave .h5 para hypertuning)
            ├── 4.KapreTunerWav_2.py (hypertuning dados obtidos em 2.featuresWave.py)
        ├── kapre (lib, para implementar extração de espectogramas como layer no modelo tensorflow, util na script 4.KapreTunerWav_2.py) 
        ├── modelsTuner 
            ├── BaselineTuner_8k_140_8.py (hypertuning de modelo)
            ├── BaselineTuner_8k_260_8.py (hypertuning de modelo)
        ├── NeSsi / tflitetools (lib, calcular caracteristicas/especificações modelo)       
    ├── MainScript_original.py (script para testes gerais )
    ├── ExportAvalicaoFinal.py (script gera csv para envio final a concurso)
    ├── AI4EDGE_1.py (resultado enviado a concurso)    
    ├── AI4EDGE_2.py (resultado enviado a concurso) 
    ├── AI4EDGE_3.py (resultado enviado a concurso)
    ├── AI4EDGE_4.py (resultado enviado a concurso)
  ├──tf23.yaml   (ficheiro de env anaconda)
`````




