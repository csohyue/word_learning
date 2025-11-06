from random import choice, random
import numpy as np

# REINFORCEMENT
LEARNING_RATE = 0.02
SMOOTHING = 0.001
THRESHOLD = 0.78

class PursuitLearner():
    """
    A learner using Pursuit
    Attributes:
        associations             Dictionary{<str> word: np.array<float> association values
        meanings                 [<str> object] index of the meaning corresponds to index of association value
    """
    def __init__(self, subject_id, retrieval=0.75):
        self.associations = {}
        self.meanings = []
        self.subject_id = subject_id
        self.retrieval_rate = retrieval

    @property
    def subject_id(self):
        """ Get subject id (as property) """
        return self.subject_id

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
                    self.associations[word] = np.append(self.associations[word], 0)

    def online_me(self, m_u):
        """
        Mutual exclusivity

        returns the best possible object associated with a word; based on which meaning is least "tied" to another word

        :param m_u: [<str>] the list of objects present in the utterance
        :return: int; index of the best meaning choice from m_u
        """
        # to keep track of the highest association for each meaning
        max_associations = []
        if len(m_u) == 0:
            return None
        for m in m_u:
            a_max = -2
            for word in self.associations:
                a_m_w = self.associations[word][self.meanings.index(m)]
                if a_m_w > a_max:
                    a_max = a_m_w
            max_associations.append(a_max)
        # get the minimum value of the max associations of the possible meanings in the utterance
        min_val = min(max_associations)
        possible_meaning_indices = []
        for i in range(len(m_u)):
            if max_associations[i] == min_val:
                possible_meaning_indices.append(self.meanings.index(m_u[i]))
        return choice(possible_meaning_indices)

    def update_association(self, word, meaning, reward):
        """
        Update association via the reinforcement algorithm if the hypothesis is confirmed

        :param word: <str> the word
        :param meaning: <str> the object in question
        :return: Nothing, mutates the association matrix
        """
        current = self.associations[word][meaning]
        if reward:
            # If the hypothesis has not yet been made, then initialize to LEARNING_RATE
            if current == 0:
                self.associations[word][meaning] = LEARNING_RATE
            # Otherwise, increment the association score
            else:
                self.associations[word][meaning] = current + LEARNING_RATE*(1 - current)
        else:
            self.associations[word][meaning] =  current * (1 - LEARNING_RATE)

    def add_novel_word(self, word, m_u):
        """
        Add a word that has yet to be encountered to the association matrix

        :param word: <str> the word
        :param m_u: [<str>] the list of objects present in the utterance
        :param weights: [<float> probability}] list of weights for objects present in the utterance
            Note that for non-HSP experiments, this will be [1, 1, 1, ...] just even weights across
        """
        meaning_i_from_me = self.online_me(m_u)
        self.associations[word] = np.zeros(len(self.meanings))
        for m in range(len(self.meanings)):
            self.associations[word][m] = 0
        if meaning_i_from_me is None:
            return None
        self.update_association(word, meaning_i_from_me, True)
        return meaning_i_from_me
        
    def get_best_meaning_i(self, word):
        """
        Gets the best hypothesis for a word given the association values probabilistically, as in test

        :param word: <str> the word
        :return: index of the object that is most associated with the word
        """
        # Select best hypothesis for a word given the association values
        max_h_list = []
        max_association = self.associations[word].max()
        for i in range(len(self.associations[word])):
            if self.associations[word][i] == max_association:
                max_h_list.append(i)
        return choice(max_h_list)

    def update_word(self, word, m_u):
        """
        Do a step of pursuit for a single word

        :param word: <str> the word
        :param m_u: [<str>] the list of objects present in the utterance
        :return: Nothing, mutates association matrix
        """
        hypothesis = self.get_best_meaning_i(word)
        if m_u == [] :
            return None
        if self.meanings[hypothesis] not in m_u:
            # Disconfirming penalizes
            self.update_association(word, hypothesis, False)
            # Select a random hypothesis from available meanings
            new_hyp = self.meanings.index(choice(m_u))
            self.update_association(word, new_hyp, True)
            return new_hyp
        else:  # if meanings[hypothesis] in m_u
            self.update_association(word, hypothesis, True)
            return hypothesis

    def train_on_utterance(self, w_u, m_u):
        """

        :param w_u: <str> the word
        :param m_u: [<str>] the list of objects present in the utterance
        """
        self.add_novel_meanings(m_u)
        selections = []
        for w in w_u:
            if w not in self.associations:
                hyp = self.add_novel_word(w, m_u)
            else:
                hyp = self.update_word(w, m_u)
            
            selections.append(hyp)
        return selections

    def generate_lexicon(self):
        """
        "Learn" words, move from association matrix to a learned lexicon
        """
        lexicon = {}
        for word in self.associations:
            lexicon[word] = []
            denominator = sum(self.associations[word]) + (len(self.meanings) * SMOOTHING)
            for meaning_i in range(self.meanings):
                numerator = self.associations[word][meaning_i] + SMOOTHING
                if numerator / denominator > THRESHOLD:
                    lexicon[word].append(self.meanings[meaning_i])
        return lexicon

    # The following functions are called externally
    def one_utterance(self, utterance):
        """
        Run a learning instance on one utterance

        :param utterance: (w_u, m_u, weights) weights optional if not hsp
        """
        w_u = utterance[0]
        if len(utterance) == 1:
            m_u = []
        else:
            m_u = utterance[1]
        hyps = self.train_on_utterance(w_u, m_u)
        return hyps

    def multiple_choice(self, word, options):
        """
        Function for doing a multiple choice question

        :param word: <str> the word
        :param options: [<str>] the objects presented as options
        :return: <str> the choice
        """
        best_hypothesis = self.get_best_meaning_i(word)
        if self.meanings[best_hypothesis] in options:
            if random() < self.retrieval_rate:
                return self.meanings[best_hypothesis]
        
        return choice(options)
        