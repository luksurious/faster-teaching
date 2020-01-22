import pickle
import random
import argparse
import time
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from concepts.letter_addition import LetterAddition
from learner_models.continuous import ContinuousModel
from learner_models.discrete import DiscreteMemoryModel
from learner_models.memoryless import MemorylessModel
from learners.human_learner import HumanLearner
from learners.sim_continuous_learner import SimContinuousLearner
from learners.sim_discrete_learner import SimDiscreteLearner
from learners.sim_memoryless_learner import SimMemorylessLearner
from plots import print_statistics_table, plot_single_errors, plot_multi_actions, plot_multi_errors, plot_multi_time, \
    plot_single_actions
from teacher import Teacher


def setup_arguments():
    parser = argparse.ArgumentParser(description='Run the "Faster Teaching via POMDP Planning" implementation using '
                                                 'simulated learners')
    # Planning arguments
    parser.add_argument('planning_model', type=str, default="memoryless",
                        choices=["memoryless", "discrete", "continuous"], nargs='?',
                        help="Which learner model to use during planning for updating the belief")
    parser.add_argument('--random', action="store_true", help='Take random actions instead of planning')
    parser.add_argument('--plan_no_noise', action="store_true", help="Disable noisy behavior in the planning models")
    parser.add_argument('--plan_discrete_memory', type=int, default=2,
                        help="Size of the memory for the planning model")
    parser.add_argument('--plan_online_horizon', type=int, default=2,
                        help="Horizon of the planning algorithm during online planning")
    parser.add_argument('--plan_online_samples', type=int, nargs='*', default=[6, 6],
                        help="Sample lens of the planning algorithm during online planning for each horizon step")
    parser.add_argument('--plan_pre_horizon', type=int, default=2, help="Horizon of the precomputed planned actions")
    parser.add_argument('--plan_pre_samples', type=int, default=6,
                        help="Number of samples per planning step for precomputed actions")
    parser.add_argument('--plan_load_actions', type=str, default=None, help="Path to file with precomputed actions")

    # Execution arguments
    parser.add_argument('-v', '--verbose', action="store_true", help="Print everything")
    parser.add_argument('-s', '--single_run', action="store_true", help="Only run one simulation")
    parser.add_argument('-c', '--sim_count', type=int, default=50, help="Number of simulations to run")
    parser.add_argument('--sim_seed', type=int, default=123, help="Base seed for the different simulation runs")
    parser.add_argument('--pool', type=int, default=None, help="Number of parallel processes to run the simulation "
                                                               "in. None (default): use number of processors "
                                                               "available. 1: no parallelization")

    # Task arguments
    parser.add_argument('-l', '--problem_len', type=int, default=6, help="Length of the letter addition problem")
    parser.add_argument('-r', '--number_range', type=int, default=6, help="Upper bound of the number range mapping")

    # Learner arguments
    parser.add_argument('-m', '--manual', action="store_true",
                        help="Manually interact with the program instead of simulation")
    parser.add_argument('--sim_model', default="memoryless", choices=["memoryless", "discrete", "continuous"],
                        help="The type of simulated learner to use")
    parser.add_argument('--sim_model_mode', default="stochastic", choices=["stochastic", "pair"],
                        help="In case of memoryless and discrete learners, you can use a stochastic mode or a pair mode"
                             " which keeps the state more similar to old states")
    parser.add_argument('--sim_discrete_memory', type=int, default=2,
                        help="Size of the memory for the simulated learner")
    parser.add_argument('--sim_pause', type=int, default=0, help="Make the simulated learner pause before continuing")
    parser.add_argument('--sim_no_noise', action="store_true", help="Disable noisy behavior in the simulated learner")

    # Teacher arguments
    parser.add_argument('--teaching_phase_actions', type=int, default=3, help="Number of teaching actions per phase")
    parser.add_argument('--max_teaching_phases', type=int, default=40,
                        help="Maximum number of teaching phases before canceling")

    return parser


def create_simulated_learner(args, concept, number_range, prior_distribution):
    if args.sim_model == 'memoryless':
        learner = SimMemorylessLearner(concept, number_range, prior_distribution)
    elif args.sim_model == 'discrete':
        learner = SimDiscreteLearner(concept, number_range, prior_distribution, args.sim_discrete_memory)
    elif args.sim_model == 'continuous':
        learner = SimContinuousLearner(concept, number_range, prior_distribution)
    else:
        raise Exception("Unknown simulation model")

    learner.pause = args.sim_pause
    learner.verbose = args.verbose
    learner.mode = args.sim_model_mode

    if args.sim_no_noise:
        learner.production_noise = 0
        learner.transition_noise = 0

    return learner


def create_belief_model(args, prior_distribution, concept):
    if args.planning_model == 'memoryless':
        belief = MemorylessModel(prior_distribution.copy(), prior_distribution, concept, verbose=args.verbose)
    elif args.planning_model == 'discrete':
        belief = DiscreteMemoryModel(prior_distribution.copy(), prior_distribution, concept,
                                     memory_size=args.plan_discrete_memory, verbose=args.verbose)
    elif args.planning_model == 'continuous':
        particle_limit = 16
        belief = ContinuousModel(prior_distribution.copy(), prior_distribution, concept, particle_limit, args.verbose)
    else:
        raise Exception("Unknown simulation model")

    if args.plan_no_noise:
        belief.transition_noise = 0
        belief.production_noise = 0

    return belief


def create_teacher(args, concept, belief):
    teacher = Teacher(concept, belief, args.random, args.teaching_phase_actions, args.max_teaching_phases)
    teacher.verbose = args.verbose
    teacher.plan_horizon = args.plan_online_horizon
    teacher.plan_samples = args.plan_online_samples

    return teacher


def create_teaching_objects(args, number_range):
    concept = LetterAddition(args.problem_len, number_range=number_range)

    prior_distribution = np.ones(len(concept.get_concept_space()))
    prior_distribution /= np.sum(prior_distribution)

    belief = create_belief_model(args, prior_distribution, concept)
    teacher = create_teacher(args, concept, belief)

    return concept, prior_distribution, belief, teacher


def setup_learner(args, concept, number_range, prior_distribution, teacher):
    if not args.manual:
        learner = create_simulated_learner(args, concept, number_range, prior_distribution)
    else:
        learner = HumanLearner(concept)
    teacher.enroll_learner(learner)

    return learner


def perform_preplanning(args, teacher):
    setup_start = time.time()
    teacher.setup(args.plan_pre_horizon, args.plan_pre_samples)

    if args.plan_pre_horizon > 0:
        print("Precomputing actions took %.2f s\n" % (time.time() - setup_start))


def handle_single_run_end(global_time_start, learner, success, teacher, model_info):
    if not success:
        teacher.reveal_answer()
        print("# Concept not learned in expected time")
        print("Last guess:")
        print(learner.concept_belief)

    print("Learning time taken: %.2f" % learner.total_time)
    print("Global time elapsed: %.2f" % (time.time() - global_time_start))
    # print(teacher.action_history)

    plot_single_errors(teacher.assessment_history, model_info)
    plot_single_actions(teacher.action_history, model_info)


def handle_multi_run_end(args, action_history, error_history, global_time_start, time_history, failures, model_info):
    plot_multi_errors(error_history, model_info)
    plt.clf()

    plot_multi_time(time_history, model_info)
    plt.clf()

    plot_multi_actions(action_history, model_info)
    plt.clf()

    print("\nLearning failures: %d/%d = %.2f%%" % (len(failures), args.sim_count, len(failures) / args.sim_count * 100))
    if len(failures) > 0:
        print("".join(["x" if i in failures else " " for i in range(args.sim_count)]) + "\n")

    print_statistics_table(error_history, time_history)
    print("Global time elapsed: %.2f" % (time.time() - global_time_start))


def describe_arguments(args):
    model_info = "Model: %s - Learner: %s - Plan: %s"
    model, learner, plan = "", "", ""

    if args.manual:
        print("Learner: Manual")
        learner = "Manual"
    else:
        if args.sim_model == 'memoryless':
            learner = "Memoryless"
            if args.sim_model_mode == 'pair':
                print("Learner: Simulated memoryless learner with pairwise updating")
                learner += " (pair)"
            else:
                print("Learner: Simulated memoryless learner with stochastic updating")
                learner += " (stoch)"
        elif args.sim_model == 'discrete':
            learner = "Discrete"
            if args.sim_model_mode == 'pair':
                print("Learner: Simulated learner with discrete memory (s=%d) and pairwise updating"
                      % args.sim_discrete_memory)
                learner += " (pair)"
            else:
                print("Learner: Simulated learner with discrete memory (s=%d) and stochastic updating"
                      % args.sim_discrete_memory)
                learner += " (stoch)"
        elif args.sim_model == 'continuous':
            print("Learner: Simulated learner with continuous memory")
            learner = "Continuous"
        if args.sim_no_noise:
            print("-- ignoring noise for simulated learners")
            learner += "(w/o noise)"

    print("")
    if args.random:
        print("Policy: Random actions")
        model = "Random"
        plan = "-"
        args.plan_pre_horizon = 0
        args.plan_online_horizon = 0
    else:
        if args.planning_model == 'memoryless':
            print("Policy: Planning actions using a memoryless belief model")
            model = "Memoryless"
        elif args.planning_model == 'discrete':
            print(
                "Policy: Planning actions using a belief model with discrete memory (s=%d)" % args.plan_discrete_memory)
            model = "Discrete"
        elif args.planning_model == 'continuous':
            print("Policy: Planning actions using a belief model with continuous memory")
            model = "Continuous"
        if args.plan_no_noise:
            print("-- ignoring noise in belief updating")
            model += " (w/o noise)"

        print("Precomputed actions: %d x %d" % (args.plan_pre_horizon, args.plan_pre_samples))
        print("Online planning: %d x %s" % (args.plan_online_horizon, args.plan_online_samples))

        plan = "%d x %d pre + %d x %s" % (args.plan_pre_horizon, args.plan_pre_samples,
                                          args.plan_online_horizon, args.plan_online_samples)

    if not args.single_run:
        print("\nSimulation: %d trials" % args.sim_count)
        learner += " x%d" % args.sim_count

    print("\nProblem: Letter Addition with %d letters, mapping to 0-%d" % (args.problem_len, args.number_range - 1))

    print("\n-------------------------\n")

    return model_info % (model, learner, plan)


def run_trial(i, args, number_range, setup=True, concept=None, prior_distribution=None, teacher=None):
    if setup:
        concept, prior_distribution, teacher = exec_setup(args, number_range, load=True,
                                                          load_file=args.plan_load_actions)
    else:
        assert concept is not None
        assert prior_distribution is not None
        assert teacher is not None
        teacher.reset()

    random.seed(args.sim_seed + i)
    np.random.seed(args.sim_seed + i)
    learner = setup_learner(args, concept, number_range, prior_distribution, teacher)

    success = teacher.teach()

    return i, success, teacher.action_history, teacher.assessment_history, learner.total_time


def exec_setup(args, number_range, load=False, load_file=None):
    random.seed(args.sim_seed)
    np.random.seed(args.sim_seed)
    concept, prior_distribution, belief, teacher = create_teaching_objects(args, number_range)

    if load:
        with open('data/actions.pickle', 'rb') as f:
            teacher.best_action_stack = pickle.load(f)
    else:
        perform_preplanning(args, teacher)

        if load_file is None:
            load_file = 'data/actions.pickle'

        with open(load_file, 'wb') as f:
            pickle.dump(teacher.best_action_stack, f)

    return concept, prior_distribution, teacher


def main():
    parser = setup_arguments()
    args = parser.parse_args()
    model_info = describe_arguments(args)

    if args.sim_count == 1:
        args.single_run = True

    number_range = list(range(0, args.number_range))

    global_time_start = time.time()

    # create objects, and pre-compute actions for all cases
    # (objects only used in single and serial execution mode)
    concept, prior_distribution, teacher = exec_setup(args, number_range, load=args.plan_load_actions is not None,
                                                      load_file=args.plan_load_actions)

    if args.single_run:
        args.verbose = True

        learner = setup_learner(args, concept, number_range, prior_distribution, teacher)

        success = teacher.teach()
        handle_single_run_end(global_time_start, learner, success, teacher, model_info)
    else:
        error_history = []
        action_history = []
        time_history = []
        failures = []

        run_parallel = args.pool != 1
        progress = None

        if run_parallel:
            # parallelize execution
            pool = Pool(processes=args.pool)
            progress = tqdm(total=args.sim_count)

            iterator = [pool.apply_async(run_trial, args=(i, args, number_range), callback=lambda x: progress.update(1))
                        for i in range(args.sim_count)]

            pool.close()

        else:
            # run on same thread
            iterator = tqdm(range(args.sim_count))

        for i in iterator:
            if run_parallel:
                i, success, single_action_history, single_assessment_history, total_time = i.get()
            else:
                i, success, single_action_history, single_assessment_history, total_time = \
                    run_trial(i, args, number_range, setup=False, concept=concept,
                              prior_distribution=prior_distribution, teacher=teacher)

            error_history.append(single_assessment_history)
            action_history.append(single_action_history)
            time_history.append(total_time)
            if not success:
                failures.append(i)

        if run_parallel and progress:
            progress.close()

        handle_multi_run_end(args, action_history, error_history, global_time_start, time_history, failures, model_info)


if __name__ == '__main__':
    main()
