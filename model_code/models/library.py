""" Module with functions for running the models """

def extract_words_meanings(situation):
    """
    Given an utterance, extract the words and objects present

    :param situation: the 2 lines in the training file
    :return: (w_u: list<str>, m_u: list<str>)
    """
    words_str, meanings_str = situation
    w_u = words_str.split()
    m_u = meanings_str.split()
    return w_u, m_u


def parse_input_data(training_data_path):
    """
    Parse the input data

    The input data is expected to have the word, then the set of meanings, and optionally,
    a set of probabilities associated with each meaning.

    :param training_data_path: str, path to training data
    :return: list [(w_u_1, m_u_1), (w_u_2, m_u_2), etc] list of word, meaning
        lists utterances
    """
    all_training = open(training_data_path).read()
    lines = all_training.strip().split('\n\n')
    parsed = []
    for line in lines:
        utterance = line.split('\n')
        words = utterance[0].split()
        meanings = utterance[1].split()
        if utterance[1] == "--":
            meanings = []
        if len(utterance) <= 2:
            parsed.append([words, meanings])
        elif len(utterance) > 2:
            probabilities = utterance[2].split()
            parsed.append([words, meanings, probabilities])
    return parsed


# Get evaluation metrics


def extract_golden_standard(path_to_standard: str = './data/train.gold') -> dict:
    """
    Extracts the golden standard associations into a dict

    :param path_to_standard: str, path to the golden standard associations
    :return: set<(word<str>, meaning<str>)> tuples
    """
    lexicon = set()
    with open(path_to_standard) as f:
        for line in f:
            word, meaning = line.split()
            lexicon.add((word, meaning))
    return lexicon


def eval_model(lexicon):
    """
    Get back the evaluation metrics on a learned lexicon

    :param lexicon: list<(word<str>, meaning<str>)> of word, meaning tuples
    :return: precision, recall, f-score
    """
    lexicon_set = set(lexicon)
    correct = len(GOLDEN_STANDARD.intersection(lexicon_set))
    p = float(correct) / len(lexicon)
    r = float(correct) / len(GOLDEN_STANDARD)
    f = (2 * p * r) / (p + r)
    return p, r, f


def eval_model_gold(lexicon, gold):
    """
    Get back the evaluation metrics on a learned lexicon

    :param lexicon: list<(word<str>, meaning<str>)> of word, meaning tuples
    :return: precision, recall, f-score
    """
    lexicon_set = set(lexicon)
    golden_standard = extract_golden_standard(gold)
    correct = len(golden_standard.intersection(lexicon_set))
    p = float(correct) / len(lexicon)
    r = float(correct) / len(golden_standard)
    if (p+r) == 0:
        f = 0
    else:
        f = (2 * p * r) / (p + r)
    return p, r, f
