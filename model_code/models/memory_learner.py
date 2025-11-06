"""Base class for MIGHT, used this for a memory-constrained Fazly et al model too """

from models.learning_space import LearningSpace


class MemoryLearner():
    """
    CSWL model with memory integrated into it
    Attributes:
        learning_space           LearningSpace object that keeps a limited number of words
        associations             Dictionary{<str> word: np.array<float> association values
        meanings                 [<str> object] index of the meaning corresponds to index 
                                of association value
        lexicon                 Dictionary{<str> word: [<str> meaning]}
    """
    def __init__(self, subject_id=0, learning_space_size=7):
        self._subject_id = subject_id
        self._learning_space = LearningSpace(learning_space_size)
        self.associations = {}
        self.meanings = []
        self.lexicon = {}
        self.removals = 0
        self.working_learning_space_size = learning_space_size

    @property
    def subject_id(self):
        """ Get subject id (as property) """
        return self._subject_id

    @property
    def learning_space(self):
        """ Get learning space size (as property) """
        return self._learning_space
