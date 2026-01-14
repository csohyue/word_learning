# word learning models

Contains the code and data for the paper:

Yue, LaTourrette, Yang, & Trueswell (submitted)

## Contents

-   `model_code` - folder containing the computational models (Section 2). Read this directory's README to see how to run the models

    -   `models` - folder containing the code for the MIGHT model (`might_learner.py`), the Pursuit model (`pursuit_learner.py`), and the Familiarity Uncertainty biased Global model (`kachergis.Rmd`), adopted from George Kachergis' (github)[<https://github.com/kachergis/word_learning_models/tree/master>].

    -   `data` - a folder containing the data `txt` files for the MIGHT and Pursuit models to run on

    -   `results` - a folder containing the model output for the MIGHT and Pursuit models

    -   `run_cswl.py` - main file for running MIGHT and Pursuit model code

-   `simulations` - folder containing the analysis code for the experimental simulations (Section 3)

    -   `simulation_analyses.html` - model simulation analysis

-   `experiments` - folder containing the experimental data and analysis files (Section 4)

    -   `expt_links.txt` - text file containing links to pre-registrations and to experiments

    -   `expt1_prereg.pdf` – pdf version of Experiment 1 preregistration

    -   `expt1_data.csv` – CSV file containing Experiment 1 data

        | column name | description |
        |-------------------------|-----------------------------------------------|
        | `experiment` | which experiment (1-4) |
        | `id` | participant id |
        | `condition` | "Confirm" or "Recall" |
        | `phase` | "learning", "test", or "debrief" |
        | `item` | order in the experiment |
        | `exposure` | which exposure (1, 2, 3, or "test") of the word |
        | `word` | novel word label |
        | `word_type` | "target", "filler", or "one-shot" |
        | `target_referent` | the referent that co-occurs with the word label |
        | `selected_object` | object referent that was clicked |
        | `target_selected` | 1 if target referent is selected, 0 otherwise |
        | `selection` | index of the selection |
        | `input` | typed input |
        | `edited_input` | cleaned input |
        | `selection_matches_guess` | 1 if the participant's clicked and typed responses match, 0 otherwise |
        | `typed_accuracy` | 1 if typed response matches the target referent, 0 otherwise |
        | `typed_superordinate_accuracy` | 1 if typed response matches the superordinate category of the target referent, 0 otherwise |

    -   `expt2_prereg.pdf` - pdf version of Experiment 2 preregistration

    -   `expt2_data.csv` - CSV file containing Experiment 2 data

        | column name       | description                                     |
        |-------------------|-------------------------------------------------|
        | `experiment`      | which experiment (1-4)                          |
        | `id`              | participant id                                  |
        | `condition`       | "Confirm" or "Recall"                           |
        | `phase`           | "learning", "test", or "debrief"                |
        | `item`            | order in the experiment                         |
        | `exposure`        | which exposure (1, 2, 3, or "test") of the word |
        | `word`            | novel word label                                |
        | `word_type`       | "target", "filler", or "one-shot"               |
        | `target_referent` | the referent that co-occurs with the word label |
        | `selected_object` | object referent that was clicked                |
        | `target_selected` | 1 if target referent is selected, 0 otherwise   |
        | `selection`       | index of the selection                          |
        | `input`           | typed input for debrief                         |

    -   `expt1_2_analysis.Rmd` - R Markdown file containing analyses for Experiments 1 and 2

    -   `expt1_2_analysis.html` - html knitted version of analyses

    -   `expt3_prereg.pdf` - pdf version of Experiment 3 preregistration

    -   `expt3_data.csv` - CSV file containing Experiment 3 data

        | column name       | description                                     |
        |-------------------|-------------------------------------------------|
        | `experiment`      | which experiment (1-4)                          |
        | `id`              | participant id                                  |
        | `condition`       | "Confirm-first" or "Conflict-first"             |
        | `phase`           | "learning", "test", or "debrief"                |
        | `item`            | order in the experiment                         |
        | `exposure`        | which exposure (1, 2, 3, or "test") of the word |
        | `word`            | novel word label                                |
        | `word_type`       | "target" or "filler"                            |
        | `selected_object` | object referent that was clicked                |
        | `target_selected` | 1 if target referent is selected, 0 otherwise   |
        | `input`           | index of the selection, input for the debrief   |

    -   `expt3_analysis.Rmd` - R Markdown file containing analyses for Experiment 3

    -   `expt3_analysis.html` - html knitted version of analyses

    -   `expt4_switch_prereg.pdf` - pdf version of Experiment 4 preregistration for switch condition

    -   `expt4_same_prereg.pdf` - pdf version of Experiment 4 preregistration for same condition

    -   `expt4_data.csv` - CSV file containing Experiment 4 data

        | column name | description |
        |-------------------|-----------------------------------------------------|
        | `experiment` | which experiment (1-4) |
        | `condition` | experimental condition: "same" or "switch" |
        | `id` | participant ID |
        | `phase` | "learning", "test", or "debrief" |
        | `block` | "pre-flush", "post-flush", or "flush" for words |
        | `word_type` | "filler" or "target" |
        | `item` | order of the trial in experiment |
        | `word` | novel word |
        | `word_i` | index of novel word item |
        | `exposure` | which exposure of the word (1, 2, 3, or test) |
        | `accuracy` | 1 if selected target referent, 0 if did not select target referent |
        | `selection` | object index selected (from pcibex) |
        | `correct1` | 1 if selected target referent at exposure 1 else 0 |
        | `correct2` | 1 if selected target referent at exposure 2 else 0 |
        | `correct3` | 1 if selected target referent at exposure 3 else 0 |
        | `value` | object referent selected |
        | `value1` | object referent selected at exposure 1 |
        | `value2` | object referent selected at exposure 2 |
        | `value3` | object referent selected at exposure 3 |

    -   `expt4_analysis.Rmd` - R Markdown file containing analyses for Experiment 4

    -   `expt4_analysis.html` - html knitted version of analyses

#### 
