""" Tune Pursuit to Rollins corpus """

from models.library import parse_input_data, eval_model
from models.pursuit_learner import PursuitLearner

def pursuit(input_path, param):
    """ Run Pursuit """
    learner = PursuitLearner(parameters=param)
    rollins = parse_input_data(input_path)
    for utterance in rollins:
        learner.one_utterance(utterance)
    lexicon = learner.generate_lexicon()
    return lexicon

def train_pursuit(iterations):
    """ Tuning Pursuit """
    best_param = []
    best_f_score, best_p, best_r = 0, 0, 0
    for learning_rate in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        for smoothing in (0.001, 0.002, 0.005, 0.01, 0.02):
            for threshold in (0.5, 0.6, 0.7, 0.75, 0.78, 0.8, 0.9):
                parameters = (learning_rate, smoothing, threshold)
                sum_p, sum_r = 0, 0
                for _ in range(iterations):
                    lexicon =  pursuit("data/pursuit/rollins.txt", parameters)
                    lexicon_list = []
                    for word, meanings in lexicon.items():
                        for meaning in meanings:
                            lexicon_list.append((word, meaning))
                    p, r, _ = eval_model(lexicon_list, "data/pursuit/gold.txt")
                    sum_p += p
                    sum_r += r
                avg_p = sum_p / iterations
                avg_r = sum_r / iterations
                if (avg_p + avg_r) == 0:
                    avg_f = 0
                else:
                    avg_f = (2 * avg_p * avg_r) / (avg_p + avg_r)
                print(parameters, avg_p, avg_r, avg_f)
                if avg_f > best_f_score:
                    best_f_score = avg_f
                    best_p = avg_p
                    best_r = avg_r
                    best_param = parameters
    print(best_f_score, best_p, best_r, best_param)


if __name__ == "__main__":
    train_pursuit(1000)
    # (0.02, 0.001, 0.78) 0.5326584358293635 0.3452647058823516 0.4189618645290679
