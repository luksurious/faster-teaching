import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import termtables as tt

from json_encoder import CustomEncoder

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


def plot_single_errors(error_history, model_subtitle=None):
    plt.plot(error_history)

    title = "Errors during assessment phase"
    add_titles(title, model_subtitle)

    plt.savefig(OUTPUT + "single-errors_%d.png" % time.time())


def plot_multi_errors(error_history, model_subtitle=None, mode='multi', finish_time=''):
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
    plt.savefig(OUTPUT + mode + "_errors_%d.png" % finish_time)


def plot_multi_time(time_history, model_subtitle=None, mode='multi', finish_time=''):
    sns.violinplot(y=time_history)

    title = "Average time to complete"
    add_titles(title, model_subtitle)

    plt.savefig(OUTPUT + mode + "_time_%d.png" % finish_time)


def plot_multi_actions(action_history, model_subtitle=None, mode='multi', finish_time=''):
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

    plt.savefig(OUTPUT + mode + "_actions_%d.png" % finish_time)


def print_statistics_table(error_history, time_history, plan_duration_history):
    learned_history = [np.argmin(errors)+1 if min(errors) == 0 else 40 for errors in error_history]

    plan_duration_history = [item for plan_durations in plan_duration_history
                             if len(plan_durations) > 0 for item in plan_durations]

    print("Online plannings done: %d" % len(plan_duration_history))

    np.set_printoptions(precision=2)
    print("\nSome statistics")
    stats_arr = [
        ["Time"] + ["%.2f" % item
                    for item in [np.mean(time_history), np.median(time_history), np.std(time_history)]],

        ["Phases"] + ["%.2f" % item
                      for item in [np.mean(learned_history), np.median(learned_history), np.std(learned_history)]],

        ["Planning duration"] + ["%.2f" % item
                                 for item in [np.mean(plan_duration_history), np.median(plan_duration_history),
                                              np.std(plan_duration_history)]]
    ]
    print(tt.to_string(stats_arr, header=["", "Mean", "Median", "SD"], alignment="lrrr"))

    return stats_arr


def add_titles(title, model_subtitle):
    plt.suptitle(title, fontsize=12)
    if model_subtitle:
        plt.title(model_subtitle, fontsize=9)


def save_raw_data(action_history, error_history, time_history, failures, stats, response_history,
                  plan_duration_history, pre_plan_duration, mode, finish_time):
    with open(OUTPUT + mode + "_data_%d.json" % finish_time, 'w') as file:
        file.write(json.dumps({
            "stats": stats,
            "precomputation": pre_plan_duration,
            "actions": action_history,
            "responses": response_history,
            "errors": error_history,
            "time": time_history,
            "plan_durations": plan_duration_history,
            "failures": failures
        }, cls=CustomEncoder))
