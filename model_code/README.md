#### models

`models` contains the code for the MIGHT model (`might_learner.py`), the Pursuit model (`pursuit_learner.py`), and the Familiarity Uncertainty biased Global model (`kachergis.Rmd`), adopted from George Kachergis' (github)[<https://github.com/kachergis/word_learning_models/tree/master>].

The MIGHT model is built on the `MemoryLearner class`, and the `learningspace` is defined in `learning_space.py` (I used this to apply the memory component on different learning strategies in the first iteration of the paper). `library.py` contains some functions that are helpful for running the model.

To run MIGHT or Pursuit, put the input data into the `data` directory, as described below.

The FUbG model takes in the data within an R file. Linking the two would be ideal, but there are some complications. Look at `model_code/models/fubg_simulations.Rmd` for an example.

#### data

`data` is a folder containing the data `txt` files.

Each directory in `data` corresponds to an experiment. If there are different conditions within the experiment, then the `{CONDITION_NAME}` prefixes each of the training and testing files. To run most simply, have a training and testing file for each condition with the following naming standard: `{CONDITION_NAME}_training.txt` and `{CONDITION_NAME}_testing.txt`.

The format for the input files is as follows – each exposure gets two lines: the labels and the referents, followed by an empty line.

```         
LABEL1 LABEL2
REFERENT1 REFERENT2 REFERENT3

LABEL2 LABEL3
REFERENT3 REFERENT2 REFERENT1
```

If the label for the target referent of a word is not exactly the same as the word label, you will need a `gold.txt` file that represents the gold standard of label-referent mappings. Each line represents a mapping: label referent (with a space between them).

```         
LABEL1 REFERENT1
LABEL2 REFERENT2
```

If there are different learning conditions and just one test, you can have a single testing file, which you'll pass in as an optional argument. If there is a different combination of learning and testing files, you can pass in a `paths.txt` file indicating the training-testing pairs for each condition (look at the `yurovsky` data for an example of both `paths.txt` and `gold.txt`).

### results

The result files of the MIGHT and Pursuit learners

#### run_cswl.py

`run_cswl.py` is the main function that you need.

usage: `computational models for cswl [-h] [-cond CONDITION] [-paths PATHS_TO_DATA] [-test TESTING_PATH] [-m MEMORY] [-c COUNT] [-gold GOLD] [-rep REPETITIONS] [-rand] model experiment`

run word learning

positional arguments:

-   `model` model (might or pursuit)

-   `experiment` experiment (should be the directory name in 'data')

optional arguments:

-   `-h`, `--help` show this help message and exit

-   `-cond`, `--condition` condition (should prefix training & testing files)

-   `-paths`, `--paths_to_data` name of txt document with training, testing pairs

-   `-test`, `--testing_path` name of testing file if it doesn't match training (default matches training)

-   `-m`, `--memory` size of learning-space for MIGHT (default 7)

-   `-c`, `--count` number of subjects (default 300)

-   `-gold`, `--gold` name of file with gold standard (default (LABEL, label))

-   `-rep`, `--repetitions` Number of repetitions, default 1

-   `-rand`, `--randomize` if should be randomized, default FALSE

For example:

```         
python run_cswl.py might expt1_2 # Runs the MIGHT model on Expt 1/2
python run_cswl.py might expt1_2 -m 11 # Runs the MIGHT model on Expt 1/2 with larger memory 
python run_cswl.py pursuit expt1_2 # Runs the Pursuit model on Expt 1/2
python run_cswl.py might yurovsky -paths paths.txt -gold gold.txt
```
