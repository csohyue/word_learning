# word learning models

Christine Soh Yue, Sandy LaTourrette, Charles Yang, and John Trueswell

## experiments

`experiments` is a folder containing the experimental data and analysis files (Section 4)

## simulations

`simulations` is a folder containing the analysis code for the experimental simulations (Section 3)

## model_code

`model_code` is a folder containing the computational models. Go into this directory to run a word learning model.

### models

`models` contains the code for the MIGHT model (`might_learner.py`), the Pursuit model (`pursuit_learner.py`), and the Familiarity Uncertainty based Global model (`kachergis.Rmd`), adopted from George Kachergis' 
(github)[https://github.com/kachergis/word_learning_models/tree/master].

The MIGHT model is built on the MemoryLearner class, and the learningspace is defined in `learning_space.py` (I used ths to apply the memory component on different learning strategies in the first iteration of the paper). `library.py` contains some functions that are helpful for running the model.

### data
`data` is a folder containing the data txt files. 

### run_cswl.py
`run_cswl.py` is the main function that you need.

usage: `computational models for cswl [-h] [-cond CONDITION] [-paths PATHS_TO_DATA] [-test TESTING_PATH] [-m MEMORY] [-c COUNT] [-gold GOLD] model experiment`

run word learning

positional arguments:
 - `model`                 model (might or pursuit)
 - `experiment`            experiment (should be the directory name in 'data')

optional arguments:
 - `-h`, `--help`            show this help message and exit
 - `-cond`, `--condition`
                        condition (should prefix training & testing files)
 - `-paths`, `--paths_to_data` 
                        name of txt document with training, testing pairs
 - `-test`, `--testing_path` 
                        name of testing file if it doesn't match training (default matches training)
 - `-m`, `--memory`
                        size of learning-space for MIGHT (default 7)
 - `-c`, `--count`
                        number of subjects (default 300)
 - `-gold`, `--gold`
                        name of file with gold standard (default (LABEL, label))