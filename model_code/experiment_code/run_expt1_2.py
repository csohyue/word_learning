import os
import pandas as pd
import numpy as np
from math import sqrt
import sys

sys.path.append('.')

from models.memory_bound_pursuit import MemoryBoundPursuitLearner
from models.memory_bound_xsit import MemoryBoundXSitLearner
from models.pursuit_learner import PursuitLearner
from models.library import parse_input_data
COLUMNS = ["condition", "subject", "item", "word_type", "phase", "accuracy", "selection",
           "buffer size", "word in lexicon", "association", "model"]


def get_chance(word, target_index, associations):
    sum_associations = 0
    for a in associations[word]:
        if a > 0:
            sum_associations += a
    return associations[word][target_index] / sum_associations

def run_one_exp(model, training_path, condition, mean_memory_size=10, count=300):
    total_size_lex = 0
    end_expt = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    learned = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    all_vals = []
    words = ["R1", "R2", "R3", "R4", "W1", "W2", "W3", "W4"]

    res_dict = {}
    for label in COLUMNS:
        res_dict[label] = []

    for i in range(count):
        memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
        if model == "pursuit":
            learner = PursuitLearner(0.75)
        elif model == "perf_pursuit":
            learner = PursuitLearner(1.0)
        elif model == "mbp":
            learner = MemoryBoundPursuitLearner(memory_size)
        parsed_input = parse_input_data(training_path)
        for utterance in parsed_input:
            learner.one_utterance(utterance)
        if model == "xsit":
            learner.end_of_learning_make_lexicon()

        this_round = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        
        for r_i in range(8):
            r = words[r_i]
            index = learner.meanings.index(r.lower()+"1")
            if model == "mbp" and r in learner.lexicon:
                learned[r_i] += 1
                this_round[r_i] = 1
            elif model == "mbp" and learner.working_learning_space.contains(r) and learner.associations[r][index] > 0:
                end_expt[r_i] += 1
                this_round[r_i] = get_chance(r, index, learner.associations)
        all_vals.append(this_round)
        if model == "mbp":
            total_size_lex += len(learner.lexicon)

        test = parse_input_data("data/recall/testing.txt")
        for [[word], all_hyp] in test:
            selection = learner.multiple_choice(word, all_hyp)
            o1 = selection[:-1]
            o2 = selection[-1]
            accuracy = o1 == word.lower() and o2 == "1"
            res_dict["condition"].append(condition)
            res_dict["subject"].append(i + 1)
            res_dict["item"].append(word)
            res_dict["phase"].append("testing")
            wt = "foil"
            if word[0] == "R":
                wt = "recall"
            elif word[0] == "W":
                wt = "one-shot"
            res_dict["word_type"].append(wt)
            res_dict["accuracy"].append(accuracy)
            res_dict["selection"].append(selection)
            if model == "mbp":
                res_dict["buffer size"].append(mean_memory_size)
                res_dict["word in lexicon"].append(1 if word in learner.lexicon else 0)
            else:
                res_dict["buffer size"].append("NA")
                res_dict["word in lexicon"].append("NA")
            res_dict["model"].append(model)
            if model != "mbp" or (word not in learner.lexicon and word in learner.associations):
                best_meaning = learner.get_best_meaning_i(word)
                to_add = learner.associations[word][best_meaning]
            else:
                to_add = -1
            res_dict["association"].append(to_add)
    
    print("target  ass:", sum(end_expt[:4]/count)/4)
    print("oneshot ass:", sum(end_expt[4:]/count)/4)
    target = np.array(all_vals)[:,:4]
    print("target  mean:", np.mean(target))
    print("target  se:  ", np.std(target)/sqrt(count))
    oneshot = np.array(all_vals)[:,4:]
    print("oneshot mean:", np.mean(oneshot))
    print("oneshot se:  ", np.std(oneshot)/sqrt(count))
    print(learned)
    print("lexicon size: ", total_size_lex / count)
    print((sum(learned[:4])/4)/count)
    print((sum(learned[4:])/4)/count)

    return res_dict

print("Confirm -- MBP -- 7")
confirm_mbp7 = run_one_exp("mbp", "data/recall/confirm_hyp.txt", "confirm", mean_memory_size=7)
print("Recall -- MBP -- 7")
recall_mbp7 = run_one_exp("mbp", "data/recall/recall_hyp.txt", "recall", mean_memory_size=7)
for item in COLUMNS:
    confirm_mbp7[item].extend(recall_mbp7[item])
for item in COLUMNS:
    mbp7_df = pd.DataFrame(confirm_mbp7)
mbp7_df.to_csv("mbp7_expt1.csv", index=False)

print("Confirm -- MBP -- 10")
confirm_mbp10 = run_one_exp("mbp", "data/recall/confirm_hyp.txt", "confirm", mean_memory_size=10)
print("Recall -- MBP -- 10")
recall_mbp10 = run_one_exp("mbp", "data/recall/recall_hyp.txt", "recall", mean_memory_size=10)
for item in COLUMNS:
    confirm_mbp10[item].extend(recall_mbp10[item])
for item in COLUMNS:
    mbp11_df = pd.DataFrame(confirm_mbp10)
mbp11_df.to_csv("mbp10_expt1.csv", index=False)

print("Confirm -- MBP -- 11")
confirm_mbp11 = run_one_exp("mbp", "data/recall/confirm_hyp.txt", "confirm", mean_memory_size=11)
print("Recall -- MBP -- 11")
recall_mbp11 = run_one_exp("mbp", "data/recall/recall_hyp.txt", "recall", mean_memory_size=11)
for item in COLUMNS:
    confirm_mbp11[item].extend(recall_mbp11[item])
for item in COLUMNS:
    mbp11_df = pd.DataFrame(confirm_mbp11)
mbp11_df.to_csv("mbp11_expt1.csv", index=False)

print("Confirm -- Pursuit -- 100")
confirm_perf_pursuit = run_one_exp("perf_pursuit", "data/recall/confirm_hyp.txt", "confirm", mean_memory_size=11)
print("Recall -- Pursuit -- 100")
recall_perf_pursuit = run_one_exp("perf_pursuit", "data/recall/recall_hyp.txt", "recall", mean_memory_size=11)
for item in COLUMNS:
    confirm_perf_pursuit[item].extend(recall_perf_pursuit[item])
for item in COLUMNS:
    perf_pursuit_df = pd.DataFrame(confirm_perf_pursuit)
perf_pursuit_df.to_csv("perf_pursuit_expt1.csv", index=False)

print("Confirm -- Pursuit -- 75")
confirm_pursuit = run_one_exp("pursuit", "data/recall/confirm_hyp.txt", "confirm", mean_memory_size=11)
print("Recall -- Pursuit -- 75")
recall_pursuit = run_one_exp("pursuit", "data/recall/recall_hyp.txt", "recall", mean_memory_size=11)
for item in COLUMNS:
    confirm_pursuit[item].extend(recall_pursuit[item])
for item in COLUMNS:
    pursuit_df = pd.DataFrame(confirm_pursuit)
pursuit_df.to_csv("pursuit_expt1.csv", index=False)
