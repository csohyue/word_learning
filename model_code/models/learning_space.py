""" Module for the space-limited learning space """

from random import choices

class LearningSpace:
    """ The class for the LearningSpace """
    def __init__(self, size):
        self.size = size
        self.word_list = []
        self.weights = {}

    def create_weights(self):
        """ Inverse weights determine how likely a word is to be deleted when full """
        weight_list = []
        for word in self.word_list:
            weight_list.append(1/self.weights[word])
        return weight_list

    def increment_weight(self, word):
        """ Weights get incremented with each exposure """
        self.weights[word] += 1

    def reset_weight(self, word):
        """ Weights are reset when the word is moved to the lexicon """
        self.weights[word] = 2

    def contains(self, word):
        """ Checks to see if the word is already in the learning space """
        return word in self.word_list

    def put(self, word):
        """ Adds word to the learning space """
        if word not in self.weights:
            self.weights[word] = 1
        self.word_list.append(word)

    def get(self):
        """ Deletes a word from the learning space, probabilistically by inverse weights """
        current_size = len(self.word_list)
        if current_size == 0:
            raise IndexError("Nothing to remove")
        # weighted deletion
        word = choices(self.word_list, weights=self.create_weights())[0]
        self.word_list.remove(word)
        self.weights.pop(word)
        return word

    def full(self):
        """ True if the learning space is full """
        return len(self.word_list) >= self.size
