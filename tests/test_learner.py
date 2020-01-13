from concepts.letter_addition import LetterAddition
from learners.sim_memoryless_learner import SimMemorylessLearner


def test_see_example():
    concept = LetterAddition(6)
    learner = SimMemorylessLearner(concept, list(range(0, 7)))

    learner.see_example(((0, 1), 10))
