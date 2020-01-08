import numpy as np
import random
from concepts.letter_addition import LetterAddition
from learners.human_learner import HumanLearner
from learners.sim_discrete_learner import SimDiscreteLearner
from learners.sim_memoryless_learner import SimMemorylessLearner
from teacher import Teacher
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import termtables as tt


SINGLE = False
VERBOSE = False

MODE_SIM = "simulation"
MODE_MANUAL = "human"
MODE = MODE_SIM

global_time_start = time.time()

times = []
error_history = []
action_history = []
time_history = []

problem_len = 6
# 0-x: direct mapping
# number_range = list(range(0, problem_len))
# 0-x+1: one extra letter
number_range = list(range(0, problem_len+1))

for i in range(50):
    random.seed(123+i)
    np.random.seed(123+i)

    concept = LetterAddition(problem_len)

    if MODE == MODE_SIM:
        # learner = SimMemorylessLearner(concept, number_range)
        learner = SimDiscreteLearner(concept, number_range, 2)
        learner.pause = 0
        learner.verbose = VERBOSE
    else:
        learner = HumanLearner(concept)

    teacher = Teacher(concept, 3, 200)
    teacher.verbose = VERBOSE

    setup_start = time.time()
    teacher.setup(0)
    if SINGLE:
        print("Setup took %.2f s" % (time.time() - setup_start))
    else:
        print(".", end="")

    teacher.enroll_learner(learner)

    if not teacher.teach():
        teacher.reveal_answer()
        print("# Concept not learned in expected time")
        print("Last guess:")
        print(learner.letter_values)

    if SINGLE:
        print("Time taken: %.2f" % learner.total_time)

        print("Global time elapsed: %.2f" % (time.time() - global_time_start))

        print(teacher.action_history)
        print(teacher.assessment_history)

        plt.plot(teacher.assessment_history)
        plt.title("Errors during assessment phase")
        plt.show()
        break

    action_history.append(teacher.action_history)
    error_history.append(teacher.assessment_history)
    time_history.append(learner.total_time)

if not SINGLE:
    # plt.plot(error_history)
    max_len = max([len(x) for x in error_history])
    plot_data = np.zeros((len(error_history), max_len))

    panda_data = pd.DataFrame()

    for i, seq in enumerate(error_history):
        plot_data[i][0:len(seq)] = seq

        panda_data = panda_data.append(pd.DataFrame({"error": plot_data[i], "step": list(range(max_len))}))

    # data = pd.DataFrame({"data": plot_data})

    # sns.lineplot(range(max_len), np.mean(plot_data, axis=0), ci='sd')
    sns.lineplot(x="step", y="error", data=panda_data, ci='sd')
    plt.title("Errors during assessment phase")
    plt.ylim(0)
    plt.xlim(0)
    plt.show()

    sns.violinplot(y=time_history)
    plt.title("Average time to complete")
    plt.show()

    learned_history = [np.argmin(errors) for errors in error_history]

    np.set_printoptions(precision=2)
    print("\nSome statistics")
    print(tt.to_string([
        ["Time"] + ["%.2f" % item
                    for item in [np.mean(time_history), np.median(time_history), np.std(time_history)]],

        ["Phases"] + ["%.2f" % item
                      for item in [np.mean(learned_history), np.median(learned_history), np.std(learned_history)]]
    ], header=["", "Mean", "Median", "SD"], alignment="lrrr"))


# teacher
# - has learner model
# - choose action: example, quiz, question with feedback
# - evaluate response
# - phases: 3 actions followed by assessment
# - estimates believed concepts distribution


# learner
# - real with interface
# - simulated
