import pandas as pd
import numpy as np
from random import choice, shuffle
import sys

sys.path.append('.')

from models.memory_bound_pursuit import MemoryBoundPursuitLearner
from models.pursuit_learner import PursuitLearner
from models.library import parse_input_data

COLUMNS = ["condition", "experiment", "subject", "word", "item", "instance", "phase", "accuracy", "selection",
           "buffer size", "word in lexicon", "association", "model"]
PPA_TRAINING = parse_input_data("data/ordering/ppa.txt")
APP_TRAINING = parse_input_data("data/ordering/app.txt")
PPA_TESTING = parse_input_data("data/ordering/ppa_testing.txt")
APP_TESTING = parse_input_data("data/ordering/app_testing.txt")


def run_testing(model, learner, index, selections, correct_answers, ppa=True):
    """
    Do one iteration of the test

    :param model
    :param learner
    :param index
    :param selections
    :return: number_correct <int>, responses [<str>]
    """
    if ppa:
        testing = PPA_TESTING
    else:
        testing = APP_TESTING
    res_dict = {}
    for label in COLUMNS:
        res_dict[label] = []
    for j in range(len(testing)):
        [[word], all_hyp] = testing[j]
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
        if int(word) <= 4:
            condition = "target"
        else:
            condition = "flush"
        res_dict["experiment"].append("PPA" if ppa else "APP")
        res_dict["condition"].append(condition)
        res_dict["subject"].append(index + 1)
        res_dict["word"].append(int(word))
        res_dict["item"].append(j + 43)
        res_dict["phase"].append("testing")
        res_dict["instance"].append("testing")
        if int(word) in correct_answers:
            res_dict["accuracy"].append(1 if answer == correct_answers[int(word)] else 0)
        else:
            res_dict["accuracy"].append(1 if answer == objects[0] else 0)
        res_dict["selection"].append(answer)
        if model == "mbp":
            res_dict["buffer size"].append(learner.working_learning_space.wls_size)
            res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
        else:
            res_dict["buffer size"].append("NA")
            res_dict["word in lexicon"].append("NA")
        res_dict["model"].append(model)
        if word in learner.associations:
            best_meaning = learner.get_best_meaning_i(word)
            to_add = learner.associations[word][best_meaning]
        else:
            to_add = -1
        res_dict["association"].append(to_add)
    return res_dict


def run_one_exp(model, count=300, mean_memory_size=7, ppa=True):
    if ppa:
        min_p, mid_p, max_p = 10, 14, 18
        min_a, max_a = 18, 22
        training = PPA_TRAINING
    else:
        min_a, max_a = 10, 14
        min_p, mid_p, max_p = 14, 18, 22
        training = APP_TRAINING
    all_runs = []
    for _ in COLUMNS:
        all_runs.append([])
    total_num_removals, total_lex_size_target = 0, 0
    
    for i in range(count):
        if model == "vanilla_pursuit":
            learner = PursuitLearner()
        elif model == "pursuit":
            learner = PursuitLearner(0.75)
        elif model == "perf_pursuit":
            learner = PursuitLearner(1.0)
        else:
            memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
            learner = MemoryBoundPursuitLearner(memory_size)
        
        selections = [0 for _ in range(42)]
        lex_size_target = 0
        
        res_dict = {}
        correct_answers = {}
        for label in COLUMNS:
            res_dict[label] = []
        for j in range(len(training)):
            if j == 23 and model == "mbp":  # At the end of the target exposures
                lex_size_target = len(learner.lexicon)
                
            if j < 10 or min_p <= j < mid_p or min_a <= j < max_a:  # Warm-up words and first P and A target words
                utterance = training[j]
            else:  # second encounter of P and flush words
                word = training[j][0]
                how_to_choose = training[j][1][:8]
                options = training[j][1][8:]
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
            res_dict["experiment"].append("PPA" if ppa else "APP")
            if int(word) <= 4:
                condition = "target"
            else:
                condition = "flush"
            res_dict["condition"].append(condition)
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
            if model == "mbp":
                res_dict["buffer size"].append(learner.working_learning_space_size)
                res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
            else:
                res_dict["buffer size"].append("NA")
                res_dict["word in lexicon"].append("NA")

            res_dict["model"].append(model)
            if word in learner.associations:
                best_meaning = learner.get_best_meaning_i(word)
                to_add = learner.associations[word][best_meaning]
            else:
                to_add = -1
            res_dict["association"].append(to_add)
        if model == "mbp":
            total_num_removals += learner.removals

        total_lex_size_target += lex_size_target

        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(res_dict[label])

        results = run_testing(model, learner, i, selections, correct_answers, ppa)
        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(results[label])
    
    print(total_num_removals / count)
    print(total_lex_size_target / count)
    return all_runs


def main():
    c = 300
    
    ppa_results_df = pd.DataFrame([[]]).drop(0)
    ppa_results = run_one_exp("mbp", count=c, mean_memory_size=7, ppa=True)
    app_results = run_one_exp("mbp", count=c, mean_memory_size=7, ppa=False)

    p_ppa_results = run_one_exp("pursuit", count=c, mean_memory_size=7, ppa=True)
    p_app_results = run_one_exp("pursuit", count=c, mean_memory_size=7, ppa=False)

    pp_ppa_results = run_one_exp("perf_pursuit", count=c, mean_memory_size=7, ppa=True)
    pp_app_results = run_one_exp("perf_pursuit", count=c, mean_memory_size=7, ppa=False)

    for item in COLUMNS:
        ppa_results[COLUMNS.index(item)].extend(app_results[COLUMNS.index(item)])
        ppa_results[COLUMNS.index(item)].extend(p_ppa_results[COLUMNS.index(item)])
        ppa_results[COLUMNS.index(item)].extend(p_app_results[COLUMNS.index(item)])
        ppa_results[COLUMNS.index(item)].extend(pp_ppa_results[COLUMNS.index(item)])
        ppa_results[COLUMNS.index(item)].extend(pp_app_results[COLUMNS.index(item)])
    for item in COLUMNS:
        ppa_results_df[item] = ppa_results[COLUMNS.index(item)]


    ppa_results_df.to_csv("./expt3_models.csv", index=False)

    # app_results_df.to_csv("./mbp_app.csv", index=False)


if __name__ == "__main__":
    main()
    