from random import choice, shuffle
from math import sqrt
import pandas as pd
import numpy as np
import sys

sys.path.append('.')

from run_cswl import run_one_exp_one_subject

ALL = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2',
       'd1', 'd2', 'e1', 'e2', 'f1', 'f2',
       'g', 'h', 'i', 'j', 'k', 'l']
GOLD = {'A': ['a1', 'a2'],
        'B': ['b1', 'b2'],
        'C': ['c1', 'c2'],
        'D': ['d1', 'd2'],
        'E': ['e1', 'e2'],
        'F': ['f1', 'f2'],
        'G': ['g'],
        'H': ['h'],
        'I': ['i'],
        'J': ['j'],
        'K': ['k'],
        'L': ['l']
        }
DOUBLE = {'A', 'B', 'C', 'D', 'E', 'F'}
DATA1 = "./data/homophones/hom_exp1.txt"
DATA2 = "./data/homophones/hom_exp2.txt"


def run_one_word(learner, word, initial_options):
    options = initial_options.copy()
    others = ALL.copy()
    for meaning in GOLD[word]:
        others.remove(meaning)
    other_options = 4 - len(initial_options)
    for i in range(other_options):
        new_word = choice(others)
        options.append(new_word)
        others.remove(new_word)
    shuffle(options)
    # print(word, options)
    answer = learner.multiple_choice(word, options)
    correct = 0
    prim_rec = None
    if answer in initial_options:
        correct = 1
        prim_rec = initial_options.index(answer)
    return correct, prim_rec


def run_one_trial_primacy(learner):
    # print('primacy')
    double_count = 0
    single_count = 0
    for word in GOLD:
        options = [GOLD[word][0]]
        answer = run_one_word(learner, word, options)[0]
        if word in DOUBLE:
            double_count += answer
        else:
            single_count += answer
    return double_count, single_count


def run_one_trial_recency(learner):
    # print('recency')
    double_count = 0
    single_count = 0
    for word in GOLD:
        options = [GOLD[word][-1]]
        answer = run_one_word(learner, word, options)[0]
        if word in DOUBLE:
            double_count += answer
        else:
            single_count += answer
    return double_count, single_count


def run_one_trial_both(learner):
    # print('both')
    double_count_prim = 0
    double_count_rec = 0
    single_count = 0
    for word in GOLD:
        options = []
        options.extend(GOLD[word])

        answer = run_one_word(learner, word, options)
        if word in DOUBLE:
            if answer[1] == 0:
                double_count_prim += answer[0]
            else:
                double_count_rec += answer[0]
        else:
            single_count += answer[0]
    return double_count_prim, double_count_rec, single_count


def initialize_arrays(count):
    arrays = []
    for i in range(7):
        arrays.append(np.zeros(count, float))
    return arrays


def run_one_exp(model, exp, data, attention, count=300):
    # print(model, exp)
    learning_model = []
    experiment = []
    condition = []
    word_type = []
    answer = []
    stddev = []
    values = []
    attentions = []
    all_runs = initialize_arrays(count)
    for i in range(count):
        learner = run_one_exp_one_subject(attention, model, data)
        d_p, s_p = run_one_trial_primacy(learner)
        d_r, s_r = run_one_trial_recency(learner)
        d_b_p, d_b_r, s_b = run_one_trial_both(learner)
        single_participant = [s_p, s_r, s_b, d_p, d_r, d_b_p, d_b_r]
        for k in range(7):
            all_runs[k][i] = single_participant[k]
    for k in range(7):
        stddev.append(all_runs[k].std())
        values.append(all_runs[k].mean())
    learning_model.extend([model] * 7)
    experiment.extend([exp] * 7)
    condition.extend(["primacy", "recency", "both"] * 2)
    condition.append("both")
    word_type.extend(["singleton"] * 3)
    word_type.extend(["homophone"] * 4)
    answer.extend(["singleton", "singleton", "singleton", "first meaning", "second meaning", "first meaning", "second meaning"])
    attentions.extend([attention] * 7)
    return learning_model, experiment, condition, word_type, answer, values, stddev, attentions


EXPERIMENTS = [DATA1, DATA2]


def run_all(a):
    results = pd.DataFrame([[]]).drop(0)
    all_experiments = [[], [], [], [], [], [], [], []]

    for model in ['pursuit', "perf_pursuit", "mbp"]:
        for e in range(2):
            data = EXPERIMENTS[e]
            one_experiment = run_one_exp(model, e+1, data, a)
            for i in range(8):
                all_experiments[i].extend(one_experiment[i])

    learning_model, experiment, condition, word_type, answer, values, stddev, attentions = all_experiments
    results["model"] = learning_model
    results["experiment"] = experiment
    results["condition"] = condition
    results["word_type"] = word_type
    results["referent"] = answer
    results["accuracy"] = np.array(values) / 6
    results["stddev"] = np.array(stddev) / 6
    results["stderr"] = np.array(stddev) / (sqrt(300))
    results["attention"] = attentions
    results.to_csv("./memoryresults/yurovsky_results_2025.csv", index=False)
    print(results)
    return results


def main():
    # Run with attention = 10 (total words = 12)
    run_all(7)


if __name__ == "__main__":
    main()
