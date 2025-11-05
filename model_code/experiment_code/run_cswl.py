import sys
import numpy as np
import pandas as pd
import argparse

sys.path.append('../pursuit')

sys.path.append('.')
from models.might import MIGHTLearner
from models.pursuit_learner import PursuitLearner
from models.library import parse_input_data

COLUMNS = ["model", "condition", "subject", "phase", "exposure", "word", 
           "selection", "accuracy", "learning_space_size"]

def learn_one_exp_one_subject(mean_memory_size, model, training_path):
    """
    Function used by other experimental runs -- this is one subject doing the learning phase for
    one experiment. Many experiments don't have the learning trajectory.
    """
    memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
    if model == "pursuit":
        learner = PursuitLearner(0.75)
    else:
        learner = MIGHTLearner(memory_size)
    parsed_input = parse_input_data(training_path)
    for utterance in parsed_input:
        learner.one_utterance(utterance)
    return learner

def add_to_list(one_subject, model, condition, subject, phase, exposure, word, selection, size):
    one_subject[COLUMNS.index("model")].append(model)
    one_subject[COLUMNS.index("condition")].append(condition)
    one_subject[COLUMNS.index("subject")].append(subject)
    one_subject[COLUMNS.index("phase")].append(phase)
    one_subject[COLUMNS.index("exposure")].append(exposure)
    one_subject[COLUMNS.index("word")].append(word)
    one_subject[COLUMNS.index("selection")].append(selection)
    one_subject[COLUMNS.index("accuracy")].append(int(word.lower() == selection.lower()))
    one_subject[COLUMNS.index("learning_space_size")].append(size)
    return one_subject

def track_one_subject(model, condition, subject, training_path, testing_path, memory_size):
    one_subject = []
    for _ in COLUMNS:
        one_subject.append([])

    if model == "pursuit":
        learner = PursuitLearner(0.75)
    else:
        learner = MIGHTLearner(memory_size)
    parsed_input = parse_input_data(training_path)
    for exposure in range(len(parsed_input)):
        utterance = parsed_input[exposure]
        selections = learner.one_utterance(utterance)
        words = utterance[0]
        for i in range(len(words)):
            word = words[i]
            selection = str(selections[i])
            one_subject = add_to_list(one_subject, model, condition, subject, "learning", 
                                      exposure, word, selection, memory_size)
    
    parsed_testing = parse_input_data(testing_path)
    for exposure in range(len(parsed_testing)):
        words, options = parsed_testing[exposure]
        for word in words:
            selection = learner.multiple_choice(word, options)
            one_subject = add_to_list(one_subject, model, condition, subject, "testing", 
                                      exposure, word, selection, memory_size)
    
    return one_subject

def run_experiment(model, condition, training_path, testing_path, mean_memory_size, count):
    memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
    all_subjects = []
    for _ in COLUMNS:
        all_subjects.append([])

    for subject_id in range(count):
        one_subject = track_one_subject(model, condition, subject_id, training_path, testing_path, memory_size)
        for item in COLUMNS:
            all_subjects[COLUMNS.index(item)].extend(one_subject[COLUMNS.index(item)])

    results = pd.DataFrame([[]]).drop(0)
    for item in COLUMNS:
        results[item] = all_subjects[COLUMNS.index(item)]
    results.to_csv(condition + "_results.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="computational models for cswl",description="run word learning")
    parser.add_argument('model', help="model (might or pursuit)")
    parser.add_argument('condition', help="experiment / condition (for documentation purposes)")
    parser.add_argument('training', help="the path to the training file")
    parser.add_argument('testing', help="the path to the testing file")
    parser.add_argument('--memory', help="size of learning-space for MIGHT (default 7)")
    parser.add_argument("--count", help="number of subjects (default 300)")

    args = parser.parse_args()
    memory = args.memory if args.memory else 7
    count = args.count if args.count else 300

    run_experiment(args.model, args.condition, args.training, args.testing, memory, count)
