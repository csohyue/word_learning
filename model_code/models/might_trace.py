"""Code """

from random import choice, choices, random
import numpy as np
from models.memory_learner import MemoryLearner

LEARNED = 100 # Large positive integer acts as a flag removing hypotheses in the lexicon in mutual exclusivity calcalations
NULL_HYPOTHESIS = -1 # Negative value acts as a flag indicating that item has not been seen
RESET = 0 # Zero acts as a flag indicating that the mapping is in the lexicon

# REINFORCEMENT
LEARNING_RATE = 0.05 # This value doesn't matter, as long as it is positive (0-1.0) given the competition-based thresholding

class MIGHTTraceLearner(MemoryLearner):
    """
    A learner using Memory-Integrated Generalized Hypothesis Testing (MIGHT)

    Pursuit with memory integrated into it
    Attributes:
        learning_space          LearningSpace object that keeps a limited number of words
        associations            Dictionary{<str> word: np.array<float> association values
        meanings                [<str> object] index of the meaning corresponds to index 
                                of association value
        lexicon                 Dictionary{<str> word: [<str> meaning]}
        #
        traces                  Dictionary{<str> word: [<str> meaning]}
                                When a word is forgotten, all its hypotheses are saved as a trace
        ###
    """

    # The following functions are used internally
    def add_novel_meanings(self, m_u):
        """ Adds novel objects

        :param m_u: [<str>] the list of objects present in the utterance
        """
        for meaning in m_u:
            # Check to see if the meaning is new
            if meaning not in self.meanings:
                self.meanings.append(meaning)
                for word in self.associations:
                    self.associations[word] = np.append(self.associations[word], NULL_HYPOTHESIS)
                for word in self.traces:
                    self.traces[word] = np.append(self.traces[word], 0)

    def remove_from_queue(self):
        """ Forget a word from memory

        The word is removed from both the memory buffer and the association matrix
        """
        word = self.learning_space.get()
        # create traces: first check existing associations
        if word in self.associations:
            hypothesis_traces = self.associations[word]
            if word not in self.traces:
                current = [0 for _ in range(len(self.meanings))]
            else:
                current = self.traces[word]
            for m in range(len(self.meanings)):
                current[m] = max(current[m], hypothesis_traces[m] * self.trace)
            self.traces[word] = current
        ###
        self.associations.pop(word, None)

    def online_me(self, m_u):
        """ Mutual exclusivity for mbp

        returns the best possible object associated with a word; based on which 
        meaning is least "tied" to another word

        :param m_u: [<str>] the list of objects present in the utterance
        :return: int; index of the best meaning choice from m_u
        """
        # to keep track of the highest association for each meaning
        max_associations = []
        if len(m_u) == 0:
            return None
        for m in m_u:
            a_max = -2
            m_index = self.meanings.index(m)
            for _, association in self.associations.items():
                a_m_w = association[m_index]
                a_max = max(a_max, a_m_w)
            # Check traces
            for _, traces in self.traces.items():
                if traces[m_index] > 0:
                    a_max = max(a_max, traces[m_index])
            ###
            for _, meanings in self.lexicon.items():
                if m in meanings:
                    a_max = max(a_max, LEARNED)
            max_associations.append(a_max)
        # get the minimum value of the max associations of the possible meanings in the utterance
        min_val = min(max_associations)
        possible_meaning_indices = []
        for i, referent in enumerate(m_u):
            if max_associations[i] == min_val:
                possible_meaning_indices.append(self.meanings.index(referent))
        return choice(possible_meaning_indices)

    def add_novel_word(self, word, m_u):
        """ Add a word that has yet to be encountered to the association matrix

        :param word: <str> the word
        :param m_u: [<str>] the list of objects present in the utterance
        """
        meaning_i_from_me = self.online_me(m_u)
        self.associations[word] = np.zeros(len(self.meanings))
        for m in range(len(self.meanings)):
            self.associations[word][m] = NULL_HYPOTHESIS
        if meaning_i_from_me is None:
            return None
        self.update_association(word, meaning_i_from_me)
        return meaning_i_from_me

    def get_best_meaning_i(self, word):
        """ Gets the best hypothesis for a word given the association values 
        probabilistically, as in test

        :param word: <str> the word
        :return: index of the object that is most associated with the word
        """
        if word in self.lexicon:
            return self.meanings.index(choice(self.lexicon[word]))

        ## Making learning the same as testing: Luce's axiom of choice
        positive_h_list = []
        weights = []
        # Weight is zeroed after being moved to the lexicon
        zero_h_list = []
        zero_weights = []
        if word in self.associations:
            for i in range(len(self.meanings)):
                if self.associations[word][i] > 0:
                    positive_h_list.append(i)
                    weights.append(self.associations[word][i])
                elif self.associations[word][i] == 0:
                    zero_h_list.append(i)
                    zero_weights.append(1)
        if word in self.traces and random() < (self.trace * sum(self.traces[word]) / LEARNING_RATE):
            for j in range(len(self.meanings)):
                if self.traces[word][j] > 0:
                    positive_h_list.append(j)
                    weights.append(self.traces[word][j])
        if len(positive_h_list) > 0:
            best_h = choices(positive_h_list, weights)[0]
        elif len(zero_h_list) > 0:
            best_h = choices(zero_h_list, zero_weights)[0]
        else:
            best_h = choice(range(len(self.meanings)))
        return best_h

    def update_association(self, word, meaning):
        """ Update association via the reinforcement algorithm if the hypothesis is confirmed

        :param word: <str> the word
        :param meaning: <str> the object in question
        :return: Nothing, mutates the association matrix
        """
        current = self.associations[word][meaning]
        # If the hypothesis has not yet been made, then initialize to LEARNING_RATE
        if current == NULL_HYPOTHESIS:
            # from trace
            if word in self.traces and self.traces[word][meaning] > 0:
                old = self.traces[word][meaning] / self.trace
                self.associations[word][meaning] = self.trace * (old + LEARNING_RATE*(1 - old)) + LEARNING_RATE
                self.traces[word][meaning] = 0
            else:
                self.associations[word][meaning] = LEARNING_RATE
            ###
            # self.associations[word][meaning] = LEARNING_RATE
        # Otherwise, increment the association score
        else:
            self.associations[word][meaning] = current + LEARNING_RATE*(1 - current)

    def update_word(self, word, m_u):
        """ Do a step of pursuit for a single word

        :param word: <str> the word
        :param m_u: [<str>] the list of objects present in the utterance
        :return: Nothing, mutates association matrix
        """
        hypothesis = self.get_best_meaning_i(word)
        if m_u == []:
            return None
        if self.meanings[hypothesis] not in m_u:
            if len(m_u) > 0:
                new_hyp = self.meanings.index(choices(m_u)[0])
                self.update_association(word, new_hyp)
                return new_hyp
            return None
        # if meanings[hypothesis] in m_u
        self.update_association(word, hypothesis)
        return hypothesis

    def train_on_utterance(self, w_u, m_u):
        """ Train on a single utterance

        :param w_u: <str> the word
        :param m_u: [<str>] the list of objects present in the utterance
        """
        self.add_novel_meanings(m_u)
        selections = []
        for w in w_u:
            if w in self.lexicon and set(self.lexicon[w]).intersection(set(m_u)):
                intersection = list(set(self.lexicon[w]).intersection(set(m_u)))
                index = choice(intersection)
                hyp = self.meanings.index(index)

            elif w not in self.associations:
                if self.learning_space.full():
                    self.remove_from_queue()
                    self.removals += 1
                self.learning_space.put(w)
                if m_u != []:
                    ### Check traces
                    if w in self.traces and random() < (self.trace * sum(self.traces[w]) / LEARNING_RATE):
                        self.associations[w] = np.zeros(len(self.meanings))
                        for m in range(len(self.meanings)):
                            self.associations[w][m] = NULL_HYPOTHESIS
                        hyp = self.update_word(w, m_u)
                    else:
                        hyp = self.add_novel_word(w, m_u)
                ###
                # hyp = self.add_novel_word(w, m_u)
                else:
                    hyp = self.add_novel_word(w, m_u)

            else:
                self.learning_space.increment_weight(w)
                hyp = self.update_word(w, m_u)

            selections.append(hyp)
        return selections

    def update_lexicon(self):
        """ "Learn" words, move from association matrix to a learned lexicon
        """
        for word, word_a in self.associations.items():
            top_score = sorted(word_a)[-1]
            closest_competitor_score = NULL_HYPOTHESIS
            if len(self.meanings) > 1:
                closest_competitor_score = sorted(word_a)[-2]
            meaning_i = None
            if closest_competitor_score in (NULL_HYPOTHESIS, RESET):
                if self.learning_space.full():
                    closest_competitor_score = LEARNING_RATE
                else:
                    closest_competitor_score = LEARNING_RATE*(1-LEARNING_RATE)
            # Must be more than 2x score of closest competitor
            if top_score > 2 * closest_competitor_score:
                meaning_i = np.where(word_a == top_score)[0][0]
            if meaning_i is not None:
                if word not in self.lexicon:
                    self.lexicon[word] = []
                self.lexicon[word].append(self.meanings[meaning_i])
                self.associations[word][meaning_i] = RESET

    # The following functions are called externally
    def one_utterance(self, utterance):
        """ Run a learning instance on one utterance

        :param utterance: (w_u, m_u)
        :param hsp: <Boolean> True if hsp, False otherwise
        """
        w_u = utterance[0]
        if len(utterance) == 1:
            m_u = []
        else:
            m_u = utterance[1]
        hyps = self.train_on_utterance(w_u, m_u)
        self.update_lexicon()
        return hyps

    def multiple_choice(self, word, options):
        """ Function for doing a multiple choice question

        :param word: <str> the word
        :param options: [<str>] the objects presented as options
        :return: <str> the choice
        """
        # the word is in the lexicon and the learned meaning is in the options
        if self.lexicon and word in self.lexicon:
            learned_meanings = set(options).intersection(set(self.lexicon[word]))
            if len(learned_meanings) > 0:
                return choice(list(learned_meanings))

        # if the word isn't in learning space, check the trace space or select randomly
        if word not in self.associations:
            if word in self.traces and random() < (self.trace * sum(self.traces[word]) / LEARNING_RATE): 
                possible_traces = []
                for meaning in options:
                    if meaning in self.meanings:
                        meaning_index = self.meanings.index(meaning)
                        possible_traces.append(self.traces[word][meaning_index])
                    else:
                        possible_traces.append(0)
                if sum(possible_traces) != 0:
                    return options[choices(range(len(possible_traces)), possible_traces)[0]]
            return choice(options)

        # otherwise: sample from the associations
        possible_associations = []
        for meaning in options:
            # No negative weights --> make the negative weight 0
            if meaning in self.meanings:
                meaning_index = self.meanings.index(meaning)
                if word in self.traces and random() < (self.trace * sum(self.traces[word]) / LEARNING_RATE):
                    possible_trace = self.traces[word][meaning_index]
                else:
                    possible_trace = 0
                association_score = max(self.associations[word][meaning_index], possible_trace)
            else:
                association_score = NULL_HYPOTHESIS

            if association_score == NULL_HYPOTHESIS:
                possible_associations.append(0)
            else:
                possible_associations.append(association_score)
        # choices doesn't like it if they're all zero, make them all weighted the same
        if sum(possible_associations) == 0.0:
            for i, _ in enumerate(possible_associations):
                possible_associations[i] = 1.0
        return options[choices(range(len(possible_associations)), possible_associations)[0]]

    def free_response(self, word):
        """ Function for doing a multiple choice question

        :param word: <str> the word
        :return: <str> the choice
        """
        # If the word is learned, just return the learned meaning
        if word in self.lexicon:
            return choice(self.lexicon[word])
        # If the word isn't in memory, return something random
        elif word not in self.associations:
            return choice(self.meanings)
        # Otherwise, check the learning space. As in the MC, select based on the weights
        else:
            return choices(self.meanings, self.associations[word])[0]
