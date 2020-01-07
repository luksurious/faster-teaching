import numpy as np
import random
from concepts.letter_addition import LetterAddition
from learners.human_learner import HumanLearner
from learners.sim_learner import SimLearner
from teacher import Teacher
import time
import matplotlib.pyplot as plt


random.seed(123)
np.random.seed(123)

global_time_start = time.time()

concept = LetterAddition(6)

learner = SimLearner(concept)
learner.pause = 0
learner.verbose = True
# learner = HumanLearner(concept)

teacher = Teacher(concept)
teacher.verbose = True

setup_start = time.time()
teacher.setup(4)
print("Setup took %.2f s" % (time.time() - setup_start))

teacher.enroll_learner(learner)

if not teacher.teach():
    teacher.reveal_answer()
    print("# Concept not learned in expected time")

print("Time taken: %.2f" % learner.total_time)

print("Global time elapsed: %.2f" % (time.time() - global_time_start))

print(teacher.action_history)
print(teacher.assessment_history)

plt.plot(teacher.assessment_history)
plt.title("Errors during assessment phase")
plt.show()


# teacher
# - has learner model
# - choose action: example, quiz, question with feedback
# - evaluate response
# - phases: 3 actions followed by assessment
# - estimates believed concepts distribution


# learner
# - real with interface
# - simulated
