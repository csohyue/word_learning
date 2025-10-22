import numpy as np
import sys

sys.path.append('../pursuit')

sys.path.append('.')
from mbp.memory_bound_pursuit import MemoryBoundPursuitLearner
from mbp.memory_bound_xsit import MemoryBoundXSitLearner
from mbp.pursuit_learner import PursuitLearner
from mbp.library import parse_input_data


def run_one_exp_one_subject(mean_memory_size, model, training_path):
    memory_size = max(1, round(np.random.normal(mean_memory_size, 1)))
    if model == "xsit":
        learner = MemoryBoundXSitLearner(10)
    elif model == "pursuit":
        learner = PursuitLearner(0.75)
    elif model == "perf_pursuit":
        learner = PursuitLearner(1.0)
    else:
        learner = MemoryBoundPursuitLearner(memory_size)
    parsed_input = parse_input_data(training_path)
    for utterance in parsed_input:
        # print(utterance)
        learner.one_utterance(utterance)
    if model == "xsit":
        learner.end_of_learning_make_lexicon()
    # elif model == "vanilla_pursuit":
    #     learner.update_probabilities()
    return learner
