import random
import numpy as np

from concepts.letter_addition import LetterAddition
from learners.sim_discrete_learner import SimDiscreteLearner


def test_3step_example():
    random.seed(123)
    np.random.seed(123)

    number_range = list(range(6))
    concept = LetterAddition(6, number_range)

    learner = SimDiscreteLearner(concept, number_range, 2)
    learner.verbose = False
    assert learner.concept_belief == [0, 1, 2, 3, 4, 5]

    learner.see_example(((0, 1), 5))  # A+B = 5
    learner.finish_action(((0, 1), 5))
    assert learner.concept_belief == [0, 5, 2, 3, 4, 1]

    learner.see_example(((1, 2), 4))  # B+C = 4
    learner.finish_action(((1, 2), 4))
    assert learner.concept_belief == [5, 0, 4, 3, 2, 1]

    learner.see_example(((2, 3), 3))  # C+D = 3
    learner.finish_action(((2, 3), 3))
    assert learner.concept_belief == [1, 4, 0, 3, 2, 5]


def test_readjustment_needed():
    random.seed(123)
    np.random.seed(123)

    number_range = list(range(7))
    concept = LetterAddition(6, number_range)

    learner = SimDiscreteLearner(concept, number_range, 2)
    learner.verbose = False
    assert learner.concept_belief == [0, 1, 2, 3, 4, 5]

    learner.see_example(((3, 4), 9))  # D+E = 9
    learner.finish_action(((3, 4), 9))
    assert learner.concept_belief == [0, 1, 2, 3, 6, 5]

    learner.see_question_question(((2, 5), None))  # C+F = ?
    learner.see_question_feedback(((2, 5), 3), False)
    learner.finish_action(((2, 5), 3))
    assert learner.concept_belief == [0, 4, 1, 3, 6, 2]

    learner.see_example(((1, 5), 5))  # B+F = 5
    learner.finish_action(((1, 5), 5))
    assert learner.concept_belief == [0, 4, 2, 3, 6, 1]

    learner.see_quiz(((2, 4), None))  # C+E = ?
    learner.finish_action(((2, 4), None))
    assert learner.concept_belief == [0, 4, 2, 3, 6, 1]

    learner.see_question_question(((1, 5), None))  # B+F = ?
    learner.see_question_feedback(((1, 5), 5), True)
    learner.finish_action(((1, 5), 5))
    assert learner.concept_belief == [0, 4, 2, 3, 6, 1]

    learner.see_example(((0, 3), 4))  # A+D = 4
    learner.finish_action(((0, 3), 4))
    assert learner.concept_belief == [0, 2, 1, 4, 5, 3]


