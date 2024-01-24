# DCASE
Repository of the R&D in the field of Detection and Classification of Acoustic Scenes and Events

Research in low-complexity ASC has attracted a lot of attention in recent years and has been included in the annual IEEE AASP Challenge on DCASE. Task 1 of the DCASE Challenge – Low-Complexity Acoustic Scene Classification – aims to promote the research around this subject by comparing different classification approaches using
TAU Urban Acoustic Scenes 2022 Mobile dataset [14] (publicly available). To ensure a good performance across different recording devices, the dataset includes data recorded and simulated with a variety of devices. As illustrated in Figure 1, the goal is to classify a test recording into one of the predefined ten acoustic scene classes using resource constrained devices. The challenge sets complexity limits modelled after Cortex-M4 devices constraints, imposing a maximum of 128K model parameters (including the zero-valued ones) and a maximum of 30 million Multiply Accumulate (MAC) operations per inference. The ultimate challenge is therefore to attain the generalization
power of state-of-the-art complex models with a low-complexity architecture.

# DCASE 2022
The TMSIC student model, trained with Response-Based KD, was submitted to the DCASE2022 Task 1 challenge. Its performance was tested with an evaluation dataset featuring new devices and data recorded in different cities. The model achieved the 11th place in a total of 48 models submitted and the 4th place in the teams ranking.
Includes info and the code used on the DCASE 2022 challenge

# DCASE 2023 
DCASE2023 challenge, the TMSIC student model was submitted, trained with the proposed RRS KD method. The model was quantized with PTQ using a dynamic-range. In the competition evaluation, the model was tested using an unseen dataset, featuring data recorded from different devices in order to test its ability to generalize and classify new data. The submitted model achieved an accuracy of 51.9% on the new unseen dataset, which is a significantly lower accuracy compared to the one obtained using the development dataset. This indicates that the model is overfit to the unbalanced development dataset and more advanced data augmentation and regularization techniques should have been employed. However, it is a performance improvement (0.3%) compared with the results obtained in DCASE2022 challenge, meaning that the proposed RRS method performed better than the Response-Based KD used in DCASE2022.


# DCASE 2024 
(Work in progress)
Includes info and the code used on the DCASE 2024 challenge
