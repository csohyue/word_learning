from models.memory_buffer import WorkingLearningSpace


class MemoryLearner(object):
    """
    CSWL model with memory integrated into it
    Attributes:
        working_learning_space   WorkingLearningSpace object that keeps a limited number of words
        associations             Dictionary{<str> word: np.array<float> association values
        meanings                 [<str> object] index of the meaning corresponds to index of association value
        lexicon                  Dictionary{<str> word: [<str> meaning]}
    """
    def __init__(self, working_learning_space_size=10):
        self.working_learning_space = WorkingLearningSpace(working_learning_space_size)
        self.associations = {}
        self.meanings = []
        self.lexicon = {}
        self.removals = 0
        self.working_learning_space_size = working_learning_space_size
