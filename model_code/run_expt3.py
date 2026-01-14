""" A module to run expt 3 """

import pandas as pd
import numpy as np

from models.might import MIGHTLearner
from models.pursuit_learner import PursuitLearner
from models.library import parse_input_data

COLUMNS = ["condition", "experiment", "subject", "word", "word_type", "item", "instance",
           "phase", "accuracy", "selection", "learning space size", "word in lexicon",
           "association", "model"]
CONFIRM_TRAINING = parse_input_data("data/expt3/confirm_learning.txt")
CONFLICT_TRAINING = parse_input_data("data/expt3/conflict_learning.txt")
CONFIRM_TESTING = parse_input_data("data/expt3/confirm_testing.txt")
CONFLICT_TESTING = parse_input_data("data/expt3/conflict_testing.txt")


def run_test_exposure(word, all_hyp, selections, learner):
    """
    Docstring for run_test_exposure
    
    :param testing_exposure: Description
    :param selections: Description
    :param learner: Description
    """
    checks = [int(h) for h in all_hyp[:24]]
    options = all_hyp[24:]
    objects = [0 for _ in range(12)]
    for i in range(12):
        check1 = int(checks[i * 2])
        check2 = int(checks[i * 2 + 1])
        if check1 == -1:
            objects[i] = options[i * 4]
        elif check2 == -1:
            objects[i] = options[i * 4 + selections[check1]]
        else:
            objects[i] = options[i * 4 + ((selections[check1] + selections[check2]) % 4)]
    answer = learner.multiple_choice(word, objects)
    return objects, answer

def run_testing(subject_id, selections, correct_answers, confirm=True):
    """
    Do one iteration of the tests

    :param subject_id [model, learner, index]
    :param selections 
    :return: number_correct <int>, responses [<str>]
    """
    model, learner, index = subject_id
    testing = CONFIRM_TESTING if confirm else CONFLICT_TESTING
    res_dict = {}
    for label in COLUMNS:
        res_dict[label] = []
    for j, line in enumerate(testing):
        [[word], all_hyp] = line
        objects, answer = run_test_exposure(word, all_hyp, selections, learner)
        res_dict["experiment"].append(3)
        res_dict["condition"].append("Confirm-first" if confirm else "Conflict-first")
        res_dict["word_type"].append("target" if int(word) <= 4 else "filler")
        res_dict["subject"].append(index + 1)
        res_dict["word"].append(int(word))
        res_dict["item"].append(j + 43)
        res_dict["phase"].append("test")
        res_dict["instance"].append("test")
        if int(word) in correct_answers:
            res_dict["accuracy"].append(1 if answer == correct_answers[int(word)] else 0)
        else:
            res_dict["accuracy"].append(1 if answer == objects[0] else 0)
        res_dict["selection"].append(answer)
        if model == "might":
            res_dict["learning space size"].append(learner.learning_space.size)
            res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
        else:
            res_dict["learning space size"].append("NA")
            res_dict["word in lexicon"].append("NA")
        res_dict["model"].append(model)
        if word in learner.associations:
            to_add = learner.associations[word][learner.get_best_meaning_i(word)]
        else:
            to_add = -1
        res_dict["association"].append(to_add)
    return res_dict

def run_one_exp(model, count=300, mean_memory_size=7, confirm=True):
    """
    Docstring for run_one_exp
    
    :param model: Description
    :param count: Description
    :param mean_memory_size: Description
    :param confirm: Description
    """
    if confirm:
        min_p, mid_p, max_p = 10, 14, 18
        min_a, max_a = 18, 22
        training = CONFIRM_TRAINING
    else:
        min_a, max_a = 10, 14
        min_p, mid_p, max_p = 14, 18, 22
        training = CONFLICT_TRAINING
    all_runs = []
    for _ in COLUMNS:
        all_runs.append([])
    total_num_removals, total_lex_size_target = 0, 0
    for i in range(count):
        if model == "pursuit":
            learner = PursuitLearner(0.75)
        elif model == "might":
            memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
            learner = MIGHTLearner(memory_size)
        else:
            memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
            learner = MIGHTLearner(memory_size)
        selections = [0 for _ in range(42)]
        res_dict = {}
        correct_answers = {}
        for label in COLUMNS:
            res_dict[label] = []
        for j, line in enumerate(training):
            # Warm-up words and first P and A target words
            if j < 10 or min_p <= j < mid_p or min_a <= j < max_a:
                utterance = line
            else:  # second encounter of P and filler words
                word = line[0]
                how_to_choose = line[1][:8]
                options = line[1][8:]
                objects = [0 for _ in range(4)]
                for k in range(4):
                    check1 = int(how_to_choose[k * 2])
                    check2 = int(how_to_choose[k * 2 + 1])
                    if check1 == -1:
                        objects[k] = options[k * 4]
                    elif check2 == -1:
                        objects[k] = options[k * 4 + selections[check1]]
                    else:
                        objects[k] = options[k * 4 + ((selections[check1] + selections[check2]) % 4)]
                utterance = [word, objects]
            selection_i = learner.one_utterance(utterance)[0]
            selections[j] = utterance[1].index(learner.meanings[selection_i])
            selection = learner.meanings[selection_i]
            word = utterance[0][0]
            res_dict["experiment"].append(3)
            res_dict["condition"].append("Confirm-first" if confirm else "Conflict-first")
            res_dict["word_type"].append("target" if int(word) <= 4 else "filler")
            res_dict["subject"].append(i + 1)
            res_dict["word"].append(int(word))
            res_dict["item"].append(j + 1)
            res_dict["phase"].append("learning")
            if j < 10:  # Warm-up 10
                res_dict["accuracy"].append(0)
                res_dict["instance"].append(1)
            elif min_p <= j < mid_p:  # First P
                res_dict["accuracy"].append(1)
                res_dict["instance"].append(1)
                correct_answers[int(word)] = selection
            elif min_a <= j < max_a:  # A
                res_dict["accuracy"].append(0)
                res_dict["instance"].append(1)
            else:
                if mid_p <= j < max_p or 22 <= j < 32:  # Second P or Flush 1
                    res_dict["instance"].append(2)
                else:
                    res_dict["instance"].append(3)
                if int(word) in correct_answers:
                    res_dict["accuracy"].append(1 if selection == correct_answers[int(word)] else 0)
                else:
                    res_dict["accuracy"].append(1 if word == selection[:-2] else 0)
            res_dict["selection"].append(selection)
            if model == "might":
                res_dict["learning space size"].append(learner.working_learning_space_size)
                res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
            else:
                res_dict["learning space size"].append("NA")
                res_dict["word in lexicon"].append("NA")

            res_dict["model"].append(model)
            if word in learner.associations:
                best_meaning = learner.get_best_meaning_i(word)
                to_add = learner.associations[word][best_meaning]
            else:
                to_add = -1
            res_dict["association"].append(to_add)
        if model == "might":
            total_num_removals += learner.removals

        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(res_dict[label])

        results = run_testing([model, learner, i], selections, correct_answers, confirm)
        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(results[label])
    print(total_num_removals / count)
    print(total_lex_size_target / count)
    return all_runs


def main():
    c = 300
    results_df = pd.DataFrame([[]]).drop(0)
    results = run_one_exp("might", count=c, mean_memory_size=7, confirm=True)
    conflict_results = run_one_exp("might", count=c, mean_memory_size=7, confirm=False)

    p_results = run_one_exp("pursuit", count=c, mean_memory_size=7, confirm=True)
    p_conflict_results = run_one_exp("pursuit", count=c, mean_memory_size=7, confirm=False)

    for item in COLUMNS:
        results[COLUMNS.index(item)].extend(conflict_results[COLUMNS.index(item)])
        results[COLUMNS.index(item)].extend(p_results[COLUMNS.index(item)])
        results[COLUMNS.index(item)].extend(p_conflict_results[COLUMNS.index(item)])
    for item in COLUMNS:
        results_df[item] = results[COLUMNS.index(item)]


    results_df.to_csv("results/expt3_models.csv", index=False)


if __name__ == "__main__":
    main()
    