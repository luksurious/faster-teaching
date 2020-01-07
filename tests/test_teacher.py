import random
import numpy as np

import pytest

from concepts.letter_addition import LetterAddition
from teacher import Teacher


def test_class_creation():
    random.seed(123)
    np.random.seed(123)
    concept = LetterAddition(6)
    teacher = Teacher(concept)

    assert isinstance(teacher, Teacher)


def test_precompute():
    # profiling with 3 levels: 3:28m

    random.seed(123)
    np.random.seed(123)
    concept = LetterAddition(6)
    # Expected letters: {'A': 6, 'B': 3, 'C': 4, 'D': 0, 'E': 2, 'F': 5}
    teacher = Teacher(concept)
    teacher.setup(0)

    # initial belief uniformly distributed
    tree = {
        "belief": teacher.belief,
        "children": []
    }
    teacher.forward_plan(tree, 3, [45]*3)
    # print(tree)
    # one level tree
    assert len(tree["children"]) == 45

    expected_items = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
    ]
    for i in range(0, len(expected_items)):
        child = tree["children"][i*3]
        assert child['item'][0] == expected_items[i]


def test_planning2():
    random.seed(123)
    np.random.seed(123)
    concept = LetterAddition(3)
    # Expected letters: {'A': 6, 'B': 3, 'C': 4, 'D': 0, 'E': 2, 'F': 5}
    teacher = Teacher(concept)
    teacher.setup(0)

    # initial belief uniformly distributed
    tree = {
        "belief": teacher.belief,
        "children": []
    }
    teacher.forward_plan(tree, 2, [9]*2)

    print("")
    print(concept.get_true_concepts())
    teacher.print_plan_tree(tree)

    print(teacher.find_optimal_action_path(tree))

