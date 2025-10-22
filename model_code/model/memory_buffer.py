from random import sample
from random import choices


class WorkingLearningSpace:
    def __init__(self, size):
        self.wls_size = size
        self.word_list = []
        self.weights = {}

    def create_weights(self):
        weight_list = []
        for word in self.word_list:
            weight_list.append(1/self.weights[word])
        return weight_list

    def increment_weight(self, word):
        self.weights[word] += 1

    def reset_weight(self, word):
        self.weights[word] = 1

    def contains(self, word):
        return word in self.word_list

    def put(self, word):
        if word not in self.weights:
            self.weights[word] = 1#2
        self.word_list.append(word)

    def get(self):
        current_size = len(self.word_list)
        if current_size == 0:
            raise IndexError("Nothing to remove")
        # weighted deletion
        word = choices(self.word_list, weights=self.create_weights())[0]
        self.word_list.remove(word)
        self.weights.pop(word)
        return word

    def full(self):
        return len(self.word_list) >= self.wls_size
