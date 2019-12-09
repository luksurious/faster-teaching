import random

import pytest

from concepts.letter_arithmetic import LetterArithmetic


def test_class_creation():
    random.seed(0)
    concept = LetterArithmetic(6)
    expected = {'A': 6, 'B': 3, 'C': 4, 'D': 0, 'E': 2, 'F': 5}

    assert concept.get_true_concepts() == expected
