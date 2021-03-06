import numpy as np

from concepts.letter_addition import LetterAddition
from random_ng import rand_ng


def test_class_creation():
    rand_ng.seed(123)

    concept = LetterAddition(6)

    # elements equal to letter with corresponding value (i.e. B = 2)
    expected = [0, 3, 1, 4, 5, 2]

    assert np.all(concept.get_true_concepts() == expected)


def test_generate_equation():
    rand_ng.seed(123)

    concept = LetterAddition(6)

    equation = concept.generate_equation(2)

    assert len(equation) == 2
    assert equation == [0, 3]


def test_evaluate_equation():
    rand_ng.seed(123)

    concept = LetterAddition(6)

    equation = concept.generate_equation(2)
    result = concept.evaluate_equation(equation)

    assert result == 4


def test_readable_format():
    rand_ng.seed(123)

    concept = LetterAddition(6)

    equation = concept.generate_equation(2)
    result = concept.evaluate_equation(equation)
    output = concept.gen_readable_format((equation, result))

    assert output == "A + D = 4"
