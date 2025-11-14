"""Module providing the code to run a Cross-Situational Word Learning task """

import argparse
import os
import numpy as np
import pandas as pd
from models.might import MIGHTLearner
from models.pursuit_learner import PursuitLearner
from models.library import parse_input_data, extract_golden_standard

COLUMNS = ["experiment", "condition", "model", # fixed for the run
           "learning_space_size", "subject", # fixed for the subject
           "phase", "trial_index", "exposure", "word", "selection", "accuracy", # for each exposure
           "previous_correct" # for each exposure
           ]

EXPOSURE_COLUMNS = COLUMNS[5:]

def learn_one_exp_one_subject(mean_memory_size, model, training):
    """
    Function used by other experimental runs -- this is one subject doing the learning phase for
    one experiment. Many experiments don't have the learning trajectory.
    """
    memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
    if model == "pursuit":
        learner = PursuitLearner(0.75)
    else:
        learner = MIGHTLearner(memory_size)
    parsed_input = parse_input_data(training)
    for utterance in parsed_input:
        learner.one_utterance(utterance)
    return learner

def add_exposure_to_log(log, values):
    """ Add a single exposure to the log
    """
    for item, value in values.items():
        log[EXPOSURE_COLUMNS.index(item)].append(value)
    return log

def get_accuracy(word, selection, path_to_gold):
    """ Get accuracy of selection """
    if path_to_gold:
        correct_pairings = extract_golden_standard(path_to_gold)
        return int((word, selection) in correct_pairings)

    return int(word.lower() == selection.lower())

def one_subject_learning(learner, training, log, gold_path):
    """ One subject, learning exposure
    
    """
    all_words_seen = {}
    previous_correct = {}
    parsed_input = parse_input_data(training)
    for index_i, utterance in enumerate(parsed_input):
        selections = learner.one_utterance(utterance)
        for i, word in enumerate(utterance[0]): # words
            if word not in all_words_seen:
                all_words_seen[word] = 0
                previous_correct[word] = None
            all_words_seen[word] += 1
            if selections[i] is not None:
                selection = str(learner.meanings[selections[i]])
                accuracy = get_accuracy(word, selection, gold_path)
            else:
                selection = None
                accuracy = None
            values = {"trial_index": index_i + 1, "exposure": all_words_seen[word],
                        "word": word, "selection": selection, 
                        "accuracy": accuracy, "previous_correct": previous_correct[word], 
                        "phase": "learning"}
            log = add_exposure_to_log(log, values)
            previous_correct[word] = accuracy
    return [learner, log, previous_correct]

def one_subject_testing(learner, testing, log, previous_correct, gold_path):
    """ One subject, testing
    """
    all_words_tested = {}
    parsed_testing = parse_input_data(testing)
    for index_i, [words, options] in enumerate(parsed_testing):
        for word in words:
            if word not in all_words_tested:
                all_words_tested[word] = 0
            all_words_tested[word] += 1
            selection = learner.multiple_choice(word, options)
            accuracy = get_accuracy(word, selection, gold_path)
            values = {"trial_index": index_i + 1, "exposure": all_words_tested[word],
                      "word": word, "selection": selection, 
                      "accuracy": accuracy, "previous_correct": previous_correct[word], 
                      "phase": "testing"}
            log = add_exposure_to_log(log, values)
    return log

def one_subject_all(learner, training_path, testing_path, gold):
    """ Run one subject with training and testing and return the learner and the
    log, created as a pandas dataframe
    """
    log = []
    for _ in EXPOSURE_COLUMNS:
        log.append([])
    learner, log, last_selection = one_subject_learning(learner, training_path, log, gold)
    log = one_subject_testing(learner, testing_path, log, last_selection, gold)
    subject_df = pd.DataFrame([[]]).drop(0)
    for item in EXPOSURE_COLUMNS:
        subject_df[item] = log[EXPOSURE_COLUMNS.index(item)]
    subject_df["subject"] = learner.subject_id
    if isinstance(learner, MIGHTLearner):
        subject_df["learning_space_size"] = learner.learning_space.size
    else:
        subject_df["learning_space_size"] = "NA"
    return learner, subject_df

def run_experiment(model, train_test, gold_path_file, mean_memory_size, run_count):
    """ Run experiment with multiple runs
    """
    expt_df = pd.DataFrame([[]]).drop(0)
    for run_id in range(run_count):
        memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
        if model == "pursuit":
            learner = PursuitLearner(run_id + 1, 0.75)
        else:
            learner = MIGHTLearner(run_id + 1, memory_size)
        learner, subject_df = one_subject_all(learner, train_test[0], train_test[1], gold_path_file)
        expt_df = pd.concat([subject_df, expt_df])
    return expt_df

def run_and_log_expt_condition(args, memory, count, condition, path_pair):
    """ function to run one condition with one model and write the csv
    """
    path_to_gold = "data/" + args.experiment + "/" + args.gold if args.gold else args.gold
    expt_log = run_experiment(args.model, path_pair, path_to_gold, memory, count)
    expt_log["experiment"] = args.experiment
    expt_log["condition"] = condition
    expt_log["model"] = args.model
    expt_log = expt_log[COLUMNS]
    return expt_log

def get_training_testing(experiment, condition_input, test):
    """ Get training and testing paths """
    # Extract all the conditions first (should be prefixing the training.txt files)
    conditions = []
    if condition_input is None:
        all_entries = os.listdir("data/" + experiment)
        for entry in all_entries:
            if entry[-12:] == "training.txt":
                conditions.append(entry[:-13])
    else:
        conditions.append(condition_input)

    # Get a dictionary of training files
    training_dictionary = {}
    for cond in conditions:
        training_dictionary[cond] = "data/" + experiment + "/" + cond + "_training.txt"

    # Create a dictionary with conditions --> (training, testing)
    path_dictionary = {}
    for cond, training in training_dictionary.items():
        test_file = test if test else cond + "_testing.txt"
        testing_path = "data/" + experiment + "/" + test_file
        path_dictionary[cond] = (training, testing_path)
    return path_dictionary

def extract_training_test(expt_dir, paths_file):
    """ Use golden standard """
    path_set = extract_golden_standard("data/" + expt_dir + "/" + paths_file)
    path_dictionary = {}
    for path_pair in path_set:
        train, test = path_pair
        condition = train + "_" + test
        training_file = "data/" + expt_dir + "/" + train + "_training.txt"
        testing_file = "data/" + expt_dir + "/" + test + "_testing.txt"
        path_dictionary[condition] = (training_file, testing_file)
    return path_dictionary

def define_arguments():
    """ Moving the arg parsing into a separate function """
    parser = argparse.ArgumentParser(prog="computational models for cswl",
                                     description="run word learning")
    parser.add_argument("model", help="model (might or pursuit)")
    parser.add_argument("experiment",
                        help="experiment (should be the directory name in 'data')")
    parser.add_argument("-cond", "--condition",
                        help="condition (should prefix training & testing files)", type=str)
    parser.add_argument("-paths", "--paths_to_data",
                        help="name of txt document with training, testing pairs", type=str)
    parser.add_argument("-test", "--testing_path",
                        help="name of testing file if it doesn't match training", type=str)
    parser.add_argument("-m", "--memory",help="size of learning-space for MIGHT (default 7)",
                        type=int)
    parser.add_argument("-c", "--count", help="number of subjects (default 300)", type=int)
    parser.add_argument("-gold", "--gold", help="name of file with gold standard",
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    arguments = define_arguments()

    mean_memory = arguments.memory if arguments.memory else 7
    runs = arguments.count if arguments.count else 300
    if arguments.paths_to_data:
        paths = extract_training_test(arguments.experiment, arguments.paths_to_data)
    else:
        paths = get_training_testing(arguments.experiment, arguments.condition,
                                     arguments.testing_path)
    print(paths)
    all_runs = pd.DataFrame([[]]).drop(0)

    for condition_name, training_testing in paths.items():
        expt = run_and_log_expt_condition(arguments, mean_memory, runs,
                                          condition_name, training_testing)
        all_runs = pd.concat([all_runs, expt])

    file_name = arguments.model
    if arguments.memory is not None:
        file_name = file_name + "_" + str(arguments.memory)
    file_name = file_name + "_" + arguments.experiment
    if arguments.condition is not None:
        file_name = file_name + "_" + arguments.condition
    file_name = "results/" + file_name + "_results.csv"
    all_runs.to_csv(file_name, index=False)
