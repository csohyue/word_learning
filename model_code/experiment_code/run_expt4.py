import pandas as pd
import numpy as np
from random import choice, shuffle
import sys

sys.path.append('.')

from model_code.models.might import MIGHTLearner
from models.pursuit_learner import PursuitLearner
from models.library import parse_input_data

COLUMNS = ["condition", "word_type", "subject", "item", "instance", "phase", "accuracy", "selection",
           "buffer size", "word in lexicon", "association", "model"]
TRAINING = parse_input_data("data/toilet 2.4/training.txt")
TESTING = parse_input_data("data/toilet 2.4/testing.txt")


def run_testing(model, learner, index, selections, same=False):
    """
    Do one iteration of the test

    :param model
    :param learner
    :param index
    :param selections
    :return: number_correct <int>, responses [<str>]
    """
    res_dict = {}
    for label in COLUMNS:
        res_dict[label] = []
    for [[word], all_hyp] in TESTING:
        checks = [int(h) for h in all_hyp[:5]]
        options = all_hyp[5:]
        objects = [0 for _ in range(9)]

        objects[0] = options[(selections[checks[0]] + selections[checks[1]]) % 4]
        if same:
            objects[0] = options[(selections[checks[1]]-1) % 4]
        objects[1] = options[4 + selections[checks[2]]]
        objects[2] = options[8]
        objects[3] = options[9]
        objects[4] = options[10 + (selections[checks[3]] + selections[checks[4]]) % 4]
        objects[5] = options[14]
        objects[6] = options[15]
        objects[7] = options[16]
        objects[8] = options[17]
        
        answer = learner.multiple_choice(word, objects)
        condition = "flush"
        if int(word) <= 4:
            condition = "preflush"
        elif int(word) > 14:
            condition = "postflush"
        res_dict["word_type"].append(condition)
        res_dict["condition"].append("same" if same else "switch")
        res_dict["subject"].append(index + 1)
        res_dict["item"].append(int(word))
        res_dict["phase"].append("testing")
        res_dict["instance"].append("testing")
        res_dict["accuracy"].append(1 if answer == objects[0] else 0)
        res_dict["selection"].append(answer)
        if model == "mbp":
            res_dict["buffer size"].append(learner.working_learning_space_size)
            res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
        else:
            res_dict["buffer size"].append("NA")
            res_dict["word in lexicon"].append("NA")
        res_dict["model"].append(model)
        if model == "mbp" and word not in learner.lexicon and word in learner.associations:
            best_meaning = learner.get_best_meaning_i(word)
            to_add = learner.associations[word][best_meaning]
        else:
            to_add = -1
        res_dict["association"].append(to_add)
    return res_dict


def run_one_exp_switch(model, count=300, mean_memory_size=10):
    all_runs = []
    for _ in COLUMNS:
        all_runs.append([])
    total_num_removals = 0
    total_preflush_lex_size = 0
    total_flush_lex_size = 0
    total_postflush_lex_size = 0

    for i in range(count):
        if model == "mbp":
            memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
            learner = MIGHTLearner(memory_size)
        elif model == "pursuit":
            learner = PursuitLearner(0.75)
        elif model == "perf_pursuit":
            learner = PursuitLearner(1.0)
        selections = [0 for _ in range(54)]
        preflush_lex_size = 0
        flush_lex_size = 0
        postflush_lex_size = 0

        res_dict = {}
        for label in COLUMNS:
            res_dict[label] = []
        for j in range(len(TRAINING)):
            if model == "mbp":
                if j == 23:
                    preflush_lex_size = len(learner.lexicon)
                elif j == 43:
                    flush_lex_size = len(learner.lexicon)
                elif j == len(TRAINING) - 1:
                    postflush_lex_size = len(learner.lexicon)
            if j < 14 or (41 < j < 46):
                utterance = TRAINING[j]
            else:
                word = TRAINING[j][0]
                how_to_choose = TRAINING[j][1][:8]
                options = TRAINING[j][1][8:]
                objects = [0 for _ in range(4)]
                for k in range(4):
                    check1 = int(how_to_choose[k*2])
                    check2 = int(how_to_choose[k*2 + 1])
                    if check1 == -1:
                        objects[k] = options[k*4]
                    elif check2 == -1:
                        objects[k] = options[k*4 + selections[check1]]
                    else:
                        objects[k] = options[k*4 + ((selections[check1] + selections[check2]) % 4)]
                utterance = [word, objects]
            selection_i = learner.one_utterance(utterance)[0]
            selections[j] = utterance[1].index(learner.meanings[selection_i])
            selection = learner.meanings[selection_i]
            word = utterance[0][0]
            condition = "flush"
            if int(word) <= 4:
                condition = "preflush"
            elif int(word) > 14:
                condition = "postflush"
            res_dict["word_type"].append(condition)
            res_dict["condition"].append("switch")
            res_dict["subject"].append(i + 1)
            res_dict["item"].append(int(word))
            res_dict["phase"].append("learning")
            if j < 14 or (41 < j < 46):
                res_dict["accuracy"].append(0)
                res_dict["instance"].append(1)
            else:
                if 13 < j < 18 or 21 < j < 32 or 45 < j < 50:
                    res_dict["instance"].append(2)
                else:
                    res_dict["instance"].append(3)
                res_dict["accuracy"].append(1 if word == selection[:-2] else 0)
            res_dict["selection"].append(selection)
            if model == "mbp":
                res_dict["buffer size"].append(learner.working_learning_space_size)
                res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
            else:
                res_dict["buffer size"].append("NA")
                res_dict["word in lexicon"].append("NA")
                
            res_dict["model"].append(model)
            if model == "mbp" and word not in learner.lexicon and word in learner.associations:
                best_meaning = learner.get_best_meaning_i(word)
                to_add = learner.associations[word][best_meaning]
            else:
                to_add = -1
            res_dict["association"].append(to_add)
        if model == "mbp":
            total_num_removals += learner.removals
        
        total_preflush_lex_size += preflush_lex_size
        total_flush_lex_size += (flush_lex_size - preflush_lex_size)
        total_postflush_lex_size += (postflush_lex_size - flush_lex_size)
        
        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(res_dict[label])
            
        results = run_testing(model, learner, i, selections, False)
        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(results[label])
        
    # print("removals          ", total_num_removals / count)
    # print("preflush lex size ", total_preflush_lex_size / count)
    # print("postflush lex size", total_flush_lex_size / count)
    # print("end lex size      ", total_postflush_lex_size / count)
    return all_runs


def run_one_exp_same(model, count=300, mean_memory_size=10):
    all_runs = []
    for _ in COLUMNS:
        all_runs.append([])
    total_num_removals = 0
    total_preflush_lex_size = 0
    total_flush_lex_size = 0
    total_postflush_lex_size = 0
    
    for i in range(count):
        memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
        if model == "mbp":
            memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
            learner = MIGHTLearner(memory_size)
        elif model == "pursuit":
            learner = PursuitLearner(0.75)
        elif model == "perf_pursuit":
            learner = PursuitLearner(1.0)
        
        selections = [0 for _ in range(54)]
        preflush_lex_size = 0
        flush_lex_size = 0
        postflush_lex_size = 0
        
        res_dict = {}
        for label in COLUMNS:
            res_dict[label] = []
        for j in range(len(TRAINING)):
            if model == "mbp":
                if j == 23:
                    preflush_lex_size = len(learner.lexicon)
                elif j == 43:
                    flush_lex_size = len(learner.lexicon)
                elif j == len(TRAINING) - 1:
                    postflush_lex_size = len(learner.lexicon)
            if j < 14 or (41 < j < 46):
                utterance = TRAINING[j]
            else:
                word = TRAINING[j][0]
                how_to_choose = TRAINING[j][1][:8]
                options = TRAINING[j][1][8:]
                objects = [0 for _ in range(4)]
                for k in range(4):
                    check1 = int(how_to_choose[k * 2])
                    check2 = int(how_to_choose[k * 2 + 1])
                    if check1 == -1:
                        objects[k] = options[k * 4]
                    elif check2 == -1:
                        objects[k] = options[k * 4 + ((selections[check1] - 1) % 4)]
                    else:
                        check = min(check1, check2)
                        objects[k] = options[k * 4 + ((selections[check] - 1) % 4)]
                utterance = [word, objects]
            selection_i = learner.one_utterance(utterance)[0]
            selections[j] = utterance[1].index(learner.meanings[selection_i])
            selection = learner.meanings[selection_i]
            word = utterance[0][0]
            condition = "flush"
            if int(word) <= 4:
                condition = "preflush"
            elif int(word) > 14:
                condition = "postflush"
            res_dict["word_type"].append(condition)
            res_dict["condition"].append("same")
            res_dict["subject"].append(i + 1)
            res_dict["item"].append(int(word))
            res_dict["phase"].append("learning")
            if j < 14 or (41 < j < 46):
                res_dict["accuracy"].append(1)
                res_dict["instance"].append(1)
            else:
                if 13 < j < 18 or 21 < j < 32 or 45 < j < 50:
                    res_dict["instance"].append(2)
                else:
                    res_dict["instance"].append(3)
                res_dict["accuracy"].append(1 if selections[j] == 0 else 0)
            res_dict["selection"].append(selection)
            if model == "mbp":
                res_dict["buffer size"].append(learner.working_learning_space_size)
                res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
            else:
                res_dict["buffer size"].append("NA")
                res_dict["word in lexicon"].append("NA")
            
            res_dict["model"].append(model)
            if model == "mbp" and word not in learner.lexicon and word in learner.associations:
                best_meaning = learner.get_best_meaning_i(word)
                to_add = learner.associations[word][best_meaning]
            else:
                to_add = -1
            res_dict["association"].append(to_add)
        if model == "mbp":
            total_num_removals += learner.removals
        total_preflush_lex_size += preflush_lex_size
        total_flush_lex_size += (flush_lex_size - preflush_lex_size)
        total_postflush_lex_size += (postflush_lex_size - flush_lex_size)
        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(res_dict[label])
        results = run_testing(model, learner, i, selections, True)
        for label in COLUMNS:
            all_runs[COLUMNS.index(label)].extend(results[label])
        
    print("removals          ", total_num_removals / count)
    print("preflush lex size ", total_preflush_lex_size / count)
    print("postflush lex size", total_flush_lex_size / count)
    print("end lex size      ", total_postflush_lex_size / count)
    
    return all_runs


def main():
    num = 300 

    toilet_results_df = pd.DataFrame([[]]).drop(0)
    mbp_switch_results = run_one_exp_switch("mbp", count=num, mean_memory_size=7)
    mbp_same_results = run_one_exp_same("mbp", count=num, mean_memory_size=7)
    p_switch_results = run_one_exp_switch("pursuit", count=num, mean_memory_size=7)
    p_same_results = run_one_exp_same("pursuit", count=num, mean_memory_size=7)
    pp_switch_results = run_one_exp_switch("perf_pursuit", count=num, mean_memory_size=7)
    pp_same_results = run_one_exp_same("perf_pursuit", count=num, mean_memory_size=7)
    for item in COLUMNS:
        mbp_switch_results[COLUMNS.index(item)].extend(mbp_same_results[COLUMNS.index(item)])
        mbp_switch_results[COLUMNS.index(item)].extend(p_switch_results[COLUMNS.index(item)])
        mbp_switch_results[COLUMNS.index(item)].extend(p_same_results[COLUMNS.index(item)])
        mbp_switch_results[COLUMNS.index(item)].extend(pp_switch_results[COLUMNS.index(item)])
        mbp_switch_results[COLUMNS.index(item)].extend(pp_same_results[COLUMNS.index(item)])

    for item in COLUMNS:
        toilet_results_df[item] = mbp_switch_results[COLUMNS.index(item)]
        
    toilet_results_df.to_csv("pcibex/toilet2.4_switch/toilet_102025.csv", index=False)
        

if __name__ == "__main__":
    main()