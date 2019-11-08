# import numpy as np
import random
from concepts.letter_arithmetic import LetterArithmetic
from teacher import Teacher

random.seed()


def create_letter_arithmetic(problem_len: int = 6):
    elements = {}
    start = ord('A')
    # TODO start at 0 or 1?
    numbers = list(range(1, problem_len + 1))
    for i in range(problem_len):
        letter = random.sample(numbers, 1)[0]
        elements[chr(start + i)] = letter
        numbers.remove(letter)

    return LetterArithmetic(elements)


concept = create_letter_arithmetic()

teacher = Teacher(concept)

if not teacher.teach():
    teacher.reveal_answer()

# teacher
# - has learner model
# - choose action: example, quiz, question with feedback
# - evaluate response
# - phases: 3 actions followed by assessment
# - estimates believed concepts distribution


# learner
# - real with interface
# - simulated
