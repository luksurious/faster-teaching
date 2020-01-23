import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import termtables as tt

OUTPUT = "data/"


def plot_single_actions(action_history, model_subtitle=None):
    action_types = [n[0].value for n in action_history]
    p1 = plt.bar(range(len(action_types)), [1 if n == 1 else 0 for n in action_types])
    p2 = plt.bar(range(len(action_types)), [1 if n == 2 else 0 for n in action_types])
    p3 = plt.bar(range(len(action_types)), [1 if n == 3 else 0 for n in action_types])

    title = "Planned actions per time step"
    add_titles(title, model_subtitle)

    plt.legend((p1[0], p2[0], p3[0]), ["Example", "Quiz", "Question"])
    plt.yticks([])
    plt.savefig(OUTPUT + "single-actions_%d.png" % time.time())
    plt.show()


def plot_single_errors(error_history, model_subtitle=None):
    plt.plot(error_history)

    title = "Errors during assessment phase"
    add_titles(title, model_subtitle)

    plt.savefig(OUTPUT + "single-errors_%d.png" % time.time())
    plt.show()


def plot_multi_errors(error_history, model_subtitle=None):
    max_len = max([len(x) for x in error_history])
    plot_data = np.zeros((len(error_history), max_len))

    panda_data = pd.DataFrame()

    for i, seq in enumerate(error_history):
        plot_data[i][0:len(seq)] = seq

        panda_data = panda_data.append(pd.DataFrame({"error": plot_data[i], "step": list(range(max_len))}))

    sns.lineplot(x="step", y="error", data=panda_data, ci='sd')

    title = "Errors during assessment phase"
    add_titles(title, model_subtitle)

    plt.ylim(0)
    plt.xlim(0)
    plt.savefig(OUTPUT + "multi-errors_%d.png" % time.time())
    plt.show()


def plot_multi_time(time_history, model_subtitle=None):
    sns.violinplot(y=time_history)

    title = "Average time to complete"
    add_titles(title, model_subtitle)

    plt.savefig(OUTPUT + "multi-time_%d.png" % time.time())
    plt.show()


def plot_multi_actions(action_history, model_subtitle=None):
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
    p2 = plt.bar(range(max_actions), action_types_2, bottom=action_types_1)
    p3 = plt.bar(range(max_actions), action_types_3, bottom=action_types_2 + action_types_1)
    plt.legend((p1[0], p2[0], p3[0]), ["Example", "Quiz", "Question"])

    title = "Planned actions per time step"
    add_titles(title, model_subtitle)

    plt.savefig(OUTPUT + "multi-actions_%d.png" % time.time())
    plt.show()


def print_statistics_table(error_history, time_history):
    learned_history = [np.argmin(errors) for errors in error_history]

    np.set_printoptions(precision=2)
    print("\nSome statistics")
    print(tt.to_string([
        ["Time"] + ["%.2f" % item
                    for item in [np.mean(time_history), np.median(time_history), np.std(time_history)]],

        ["Phases"] + ["%.2f" % item
                      for item in [np.mean(learned_history), np.median(learned_history), np.std(learned_history)]]
    ], header=["", "Mean", "Median", "SD"], alignment="lrrr"))


def add_titles(title, model_subtitle):
    plt.suptitle(title, fontsize=12)
    if model_subtitle:
        plt.title(model_subtitle, fontsize=9)