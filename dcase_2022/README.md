# DCASE 2022

The TMSIC student model, trained with Response-Based KD, was submitted to the
DCASE2022 Task 1 challenge. Its performance was tested with an evaluation dataset
featuring new devices and data recorded in different cities. The model achieved the
11th place in a total of 48 models submitted and the 4th place in the teams ranking
(many of these models are variations of a single proposal, as each team can submit
up to 4 models). Table 12 shows the results obtained by the 4 best classified teams.
We can see that the best classified has a superior accuracy and LogLoss, at the cost
of complexity (more parameters and MAC). Indeed, the presented model did not take
full advantage of the complex limits imposed by the challenge - the winning model
was pretty close to use the full 128K parameters and 30MMACs. Nevertheless, results
are still impressive given the model’s limited complexity, showcasing the merits of the
proposed TEOvATMSIC teacher’s ability to learn and to teach the student.
