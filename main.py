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


SINGLE = True
VERBOSE = True

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
number_range = list(range(0, problem_len))

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

    teacher = Teacher(concept, 3, 40)
    teacher.verbose = VERBOSE

    setup_start = time.time()
    teacher.setup(2, 6)

    # New data:
    # - 3*6
    #   448.44 s (observation sampling)
    #   180.21 s (all observations and skip 0; no epsilon)
    #   344.21 s (all observations, with epsilon) - no convergence - deteriorated to quizzes

    # with 10 samples
    # 3: 2s
    # 4: 17s
    # 5: 190s (~*10)
    # 6: ~30min
    # 7: ~5h
    # 8: ~2d
    # 9: ~20d

    # with 9 samples
    # 3: 1.4s
    # 4: 12s
    # 5: 117s (~*9)
    # 6: ~18min
    # 7: ~2.6h
    # 8: ~1d
    # 9: ~10d

    # with 8 samples
    # 3: 0.9s
    # 4: 7.3s
    # 5: 68s (~*8)
    # 6: ~8min
    # 7: ~1h
    # 8: ~8h
    # 9: ~3d

    # with 7 samples
    # 3: 0.6s
    # 4: 4.5s
    # 5: 33s (~*7)
    # 6: ~3.5min
    # 7: ~25min
    # 8: ~3h
    # 9: ~21h

    # with 6 samples
    # 3: 0.5s
    # 4: 2.5s
    # 5: 14s (~*6)
    # 6: ~1.2min
    # 7: ~7min
    # 8: ~42min
    # 9: ~4h

    if SINGLE:
        print("Setup took %.2f s" % (time.time() - setup_start))
    else:
        print(".", end="")

    teacher.enroll_learner(learner)

    if not teacher.teach():
        teacher.reveal_answer()
        print("# Concept not learned in expected time")
        print("Last guess:")
        print(learner.concept_belief)

    if SINGLE:
        print("Time taken: %.2f" % learner.total_time)

        print("Global time elapsed: %.2f" % (time.time() - global_time_start))

        print(teacher.action_history)
        print(teacher.assessment_history)

        plt.plot(teacher.assessment_history)
        plt.title("Errors during assessment phase")
        plt.show()

        action_types = [n[0].value for n in teacher.action_history]
        p1 = plt.bar(range(len(action_types)), [1 if n == 1 else 0 for n in action_types])
        p2 = plt.bar(range(len(action_types)), [1 if n == 2 else 0 for n in action_types])
        p3 = plt.bar(range(len(action_types)), [1 if n == 3 else 0 for n in action_types])
        plt.legend((p1[0], p2[0], p3[0]), ["Example", "Quiz", "Question"])
        plt.yticks([])
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

    max_actions = max([len(x) for x in action_history])
    action_types_1 = np.zeros(max_actions)
    action_types_2 = np.zeros(max_actions)
    action_types_3 = np.zeros(max_actions)
    for el in action_history:
        for i, v in enumerate(el):
            if v[0].value == 1:
                action_types_1[i] += 1
            elif v[0].value == 2:
                action_types_2[i] += 1
            elif v[0].value == 3:
                action_types_3[i] += 1

    p1 = plt.bar(range(max_actions), action_types_1)
    p2 = plt.bar(range(max_actions), action_types_2)
    p3 = plt.bar(range(max_actions), action_types_3)
    plt.legend((p1[0], p2[0], p3[0]), ["Example", "Quiz", "Question"])
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


# Notes:
# - Planning does not return useful actions; sampling issue? (no examples samples);
#   should it sample equations and then check all types? (although would not strictly be correct)
# - Belief update questions
# - Belief update does not scale in planning
