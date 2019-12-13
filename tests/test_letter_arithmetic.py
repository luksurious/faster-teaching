import random

import pytest

from concepts.letter_arithmetic import LetterArithmetic


def test_class_creation():
    random.seed(0)
    concept = LetterArithmetic(6)
    expected = {'A': 3, 'B': 4, 'C': 0, 'D': 2, 'E': 5, 'F': 1}

    assert concept.get_true_concepts() == expected