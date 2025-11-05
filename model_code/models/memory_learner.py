from models.learning_space import LearningSpace


class MemoryLearner(object):
    """
    CSWL model with memory integrated into it
    Attributes:
        learning_space           LearningSpace object that keeps a limited number of words
        associations             Dictionary{<str> word: np.array<float> association values
        meanings                 [<str> object] index of the meaning corresponds to index of association value
        lexicon                  Dictionary{<str> word: [<str> meaning]}
    """
    def __init__(self, learning_space_size=10):
        self.learning_space = LearningSpace(learning_space_size)
        self.associations = {}
        self.meanings = []
        self.lexicon = {}
        self.removals = 0
        self.working_learning_space_size = learning_space_size
