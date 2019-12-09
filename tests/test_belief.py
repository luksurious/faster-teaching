import random

import numpy as np

from actions import Actions
from belief import Belief
from concepts.concept_base import ConceptBase, ActionResult
from concepts.letter_arithmetic import LetterArithmetic


def test_belief_trivial():
    random.seed(0)
    concept = LetterArithmetic(2)
    assert concept.get_true_concepts() == {'A': 1, 'B': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[.5, .5],
                             action=Actions.EXAMPLE,
                             true=1,
                             response=None,
                             expected=[.5, .5])


def test_belief_three_example():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=3,
                             response=None,
                             expected=[0., 0., 0., 0.5, 0., 0.5])


def test_belief_three_example_2iter():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=3,
                             response=None,
                             expected=[0., 0., 0., 0.5, 0., 0.5])

    # memoryless model: not taking previous action into account
    check_update_belief_with(belief,
                             equation=['A', '+', 'C'],
                             cur_belief=[0., 0., 0., 0.5, 0., 0.5],
                             action=Actions.EXAMPLE,
                             true=1,
                             response=None,
                             expected=[0., 0.5, 0., 0.5, 0., 0.])


def test_belief_three_quiz_correct_with_uniform_prior():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.QUIZ,
                             true=3,
                             response=3,
                             expected=[0., 0., 0., 0.5, 0., 0.5])


def test_belief_three_quiz_false_with_uniform_prior():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.QUIZ,
                             true=3,
                             response=1,
                             expected=[.5, 0., .5, 0., 0., 0.])


def test_belief_three_quiz_invalid():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.QUIZ,
                             true=3,
                             response=5,
                             # reset
                             expected=[1 / 6 for _ in range(6)])


def test_belief_three_example_quiz_correct():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=3,
                             response=None,
                             expected=[0., 0., 0., 0.5, 0., 0.5])

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[0., 0., 0., 0.5, 0., 0.5],
                             action=Actions.QUIZ,
                             true=3,
                             response=3,
                             expected=[0., 0., 0., 0.5, 0., 0.5])


def test_belief_three_example_quiz_inconsistent():
    random.seed(0)
    concept = LetterArithmetic(3)

    assert concept.get_true_concepts() == {'A': 1, 'B': 2, 'C': 0}

    belief = create_test_belief(concept)

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[1 / 6 for _ in range(6)],
                             action=Actions.EXAMPLE,
                             true=3,
                             response=None,
                             expected=[0., 0., 0., 0.5, 0., 0.5])

    check_update_belief_with(belief,
                             equation=['A', '+', 'B'],
                             cur_belief=[0., 0., 0., 0.5, 0., 0.5],
                             action=Actions.QUIZ,
                             true=3,
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



