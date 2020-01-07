import random

import numpy as np

from actions import Actions
from belief import Belief
from concepts.concept_base import ConceptBase, ActionResult
from concepts.letter_addition import LetterAddition


def test_belief_trivial():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(2)
    assert np.all(concept.get_true_concepts() == [0, 1])

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # = A + B
                             cur_belief=[.5, .5],
                             action=Actions.EXAMPLE,
                             true=1,
                             response=None,
                             expected=[.5, .5])


def test_belief_three_example():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=2,
                             response=None,
                             expected=[0., 0.5, 0., 0., 0.5, 0.])


def test_belief_three_example_2iter():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=2,
                             response=None,
                             expected=[0., 0.5, 0., 0., 0.5, 0.])

    # memoryless model: not taking previous action into account
    check_update_belief_with(belief,
                             equation=(0, 2),  # A + C
                             cur_belief=[0., 0.5, 0., 0., 0.5, 0.],
                             action=Actions.EXAMPLE,
                             true=1,
                             response=None,
                             # expected=[0., 0.5, 0., 0.5, 0., 0.])
                             expected=[0., 1., 0., 0., 0., 0.])


def test_belief_three_quiz_correct_with_uniform_prior():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.QUIZ,
                             true=2,
                             response=2,
                             expected=[0., 0.5, 0., 0., 0.5, 0.])


def test_belief_three_quiz_false_with_uniform_prior():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.QUIZ,
                             true=2,
                             response=1,
                             expected=[.5, 0., .5, 0., 0., 0.])


def test_belief_three_quiz_invalid():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.QUIZ,
                             true=1,
                             response=5,
                             # reset
                             expected=[1 / 6 for _ in range(6)])


def test_belief_three_example_quiz_correct():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=2,
                             response=None,
                             expected=[0., 0.5, 0., 0., 0.5, 0.])

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[0., 0.5, 0., 0., 0.5, 0.],
                             action=Actions.QUIZ,
                             true=2,
                             response=2,
                             expected=[0., 0.5, 0., 0., 0.5, 0.])


def test_belief_three_example_quiz_inconsistent():
    random.seed(123)
    np.random.seed(123)

    concept = LetterAddition(3)

    assert np.all(concept.get_true_concepts() == [0, 2, 1])  # A = 0, B = 2, C = 1

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=2,
                             response=None,
                             expected=[0., 0.5, 0., 0., 0.5, 0.])

    check_update_belief_with(belief,
                             equation=(0, 1),  # A + B
                             cur_belief=[0., 0.5, 0., 0., 0.5, 0.],
                             action=Actions.QUIZ,
                             true=2,
                             response=1,
                             # ignore cur belief
                             expected=[.5, 0., .5, 0., 0., 0.])


def check_update_belief_with(belief, cur_belief, action, true, response, equation, expected):
    assert np.allclose(belief.belief_state, np.array(cur_belief))
    belief.update_belief(action, [equation, true], response)
    assert np.allclose(belief.belief_state, np.array(expected))


def create_test_belief(concept, cur_belief=None):
    concept_count = len(concept.get_concept_space())
    prior = np.array([1 / concept_count for _ in range(concept_count)])

    if cur_belief is None:
        cur_belief = prior.copy()

    belief = Belief(cur_belief, prior, concept)
    belief.transition_noise = 0
    belief.production_noise = 0
    return belief



