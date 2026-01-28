import os
import pandas as pd
import random
import string

from run_cswl import learn_one_exp_one_subject


def generate_options(meaning, all_opt, num_opt):
    options = [meaning]
    for i in range(num_opt - 1):
        o = random.choice(all_opt)
        while o in options:
            o = random.choice(all_opt)
        options.append(o)
    return options


def run_testing_generated(words, model, num_opt):
    number_correct = []
    responses = []
    a = string.ascii_uppercase[:18]
    all_opt = [b for b in a]

    for word in words:
        meaning = word.upper()
        options = generate_options(meaning, all_opt, num_opt)
        answer = model.multiple_choice(word, options)
        responses.append(answer)
        number_correct.append(1 if answer == meaning else 0)

    return number_correct, responses


def run_one_exp(model, training_path, count, num_opt=4, mean_memory_size=7):
    print(model)
    condition = training_path[-14]
    model_col = []
    words_col = []
    meanings_col = []
    response_col = []
    sample_col = []
    accuracy_col = []
    condition_col = []

    for i in range(count):
        learner = learn_one_exp_one_subject(mean_memory_size, model, training_path)
        all_words = [a for a in string.ascii_lowercase[:18]]
        accuracy, responses = run_testing_generated(all_words, learner, num_opt)
        model_col.extend([model]*len(all_words))
        words_col.extend(all_words)
        meanings_col.extend([a for a in string.ascii_uppercase[:18]])
        response_col.extend(responses)
        sample_col.extend([i]*len(all_words))
        accuracy_col.extend(accuracy)
        condition_col.extend([condition]*len(all_words))
    return [sample_col, model_col, condition_col, words_col, meanings_col, response_col, accuracy_col]


def get_all_experiments(path_to_directory):
    all_experiments = []
    for filename in os.listdir(path_to_directory):
        if "2x2_training" in filename or "3x3" in filename or "4x4" in filename:
            all_experiments.append(path_to_directory + "/" + filename)
    return all_experiments


COLUMNS = ["sample", "model", "condition", "words", "meanings", "response", "accuracy"]


def run_all_exp(directory, count=300):
    results = pd.DataFrame([[]]).drop(0)
    all_exp = get_all_experiments(directory)
    all_runs = []
    for _ in COLUMNS:
        all_runs.append([])
    for model in ["mbp", "pursuit"]:
        for exp in all_exp:
            exp_id = exp[-14]
            print(exp_id)
            cols = run_one_exp(model, exp, count=count, num_opt=int(exp_id[0]))
            for i in range(len(COLUMNS)):
                all_runs[i].extend(cols[i])
    for item in COLUMNS:
        results[item] = all_runs[COLUMNS.index(item)]
    results.to_csv("./results/" + directory.split('/')[-1] + "results.csv", index=False)
    # print(results)
    return results


def main():
    run_all_exp("data/yu_smith", count=1000)


if __name__ == "__main__":
    main()
