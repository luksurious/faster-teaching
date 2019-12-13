import numpy as np
import random
from concepts.letter_addition import LetterAddition
from teacher import Teacher

random.seed(123)
np.random.seed(123)

concept = LetterAddition(6)

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
