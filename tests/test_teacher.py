import random

import pytest

from concepts.letter_arithmetic import LetterArithmetic
from teacher import Teacher


def test_class_creation():
    random.seed(0)
    concept = LetterArithmetic(6)
    teacher = Teacher(concept)

    assert isinstance(teacher, Teacher)


def test_precompute():
    random.seed(0)
    concept = LetterArithmetic(6)
    # Expected letters: {'A': 6, 'B': 3, 'C': 4, 'D': 0, 'E': 2, 'F': 5}
    teacher = Teacher(concept)

    # initial belief uniformly distributed
    tree = teacher.precompute_actions(2)
    print(tree)
    # one level tree
    assert len(tree["children"]) == 15

    expected_items = [
        ['A', '+', 'B'],
        ['A', '+', 'C'],
        ['A', '+', 'D'],
        ['A', '+', 'E'],
        ['A', '+', 'F'],
        ['B', '+', 'C'],
        ['B', '+', 'D'],
        ['B', '+', 'E'],
        ['B', '+', 'F'],
        ['C', '+', 'D'],
        ['C', '+', 'E'],
        ['C', '+', 'F'],
        ['D', '+', 'E'],
        ['D', '+', 'F'],
        ['E', '+', 'F'],
    ]
    for i in range(len(tree["children"])):
        child = tree["children"][i]
        assert child['item'][0] == expected_items[i]
