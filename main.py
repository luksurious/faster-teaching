import random
import argparse
import time

from concepts.letter_addition import LetterAddition
from learner_models.discrete import DiscreteMemoryModel
from learner_models.memoryless import MemorylessModel
from learners.human_learner import HumanLearner
from learners.sim_continuous_learner import SimContinuousLearner
from learners.sim_discrete_learner import SimDiscreteLearner
from learners.sim_memoryless_learner import SimMemorylessLearner
from plots import *
from teacher import Teacher


def setup_arguments():
    parser = argparse.ArgumentParser(description='Run the "Faster Teaching via POMDP Planning" implementation using '
                                                 'simulated learners')
    # Planning arguments
    parser.add_argument('policy', type=str, default="plan", choices=["plan", "random"], nargs='?',
                        help='Select the policy of the teacher: "plan" (default) or "random"')
    parser.add_argument('planning_model', type=str, default="memoryless",
                        choices=["memoryless", "discrete", "continuous"], nargs='?',
                        help="Which learner model to use during planning for updating the belief")
    parser.add_argument('--plan_no_noise', action="store_true", help="Disable noisy behavior in the planning models")
    parser.add_argument('--plan_discrete_memory', type=int, default=2,
                        help="Size of the memory for the planning model")

    # Execution arguments
    parser.add_argument('-v', '--verbose', action="store_true", help="Print everything")
    parser.add_argument('-s', '--single_run', action="store_true", help="Only run one simulation")
    parser.add_argument('-c', '--sim_count', type=int, default=50, help="Number of simulations to run")
    parser.add_argument('--sim_seed', type=int, default=123, help="Base seed for the different simulation runs")

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
    parser.add_argument('--plan_online_horizon', type=int, default=2,
                        help="Horizon of the planning algorithm during online planning")
    parser.add_argument('--plan_online_samples', type=int, nargs='*', default=[6, 6],
                        help="Sample lens of the planning algorithm during online planning for each horizon step")
    parser.add_argument('--plan_pre_horizon', type=int, default=2, help="Horizon of the precomputed planned actions")
    parser.add_argument('--plan_pre_samples', type=int, default=6,
                        help="Number of samples per planning step for precomputed actions")

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
        raise Exception("Not yet implemented")
    else:
        raise Exception("Unknown simulation model")

    if args.plan_no_noise:
        belief.transition_noise = 0
        belief.production_noise = 0

    return belief


def create_teacher(args, concept, belief, learner, single_run):
    teacher = Teacher(concept, belief, args.policy, args.teaching_phase_actions, args.max_teaching_phases)
    teacher.verbose = args.verbose
    teacher.plan_horizon = args.plan_online_horizon
    teacher.plan_samples = args.plan_online_samples

    setup_start = time.time()
    teacher.setup(args.plan_pre_horizon, args.plan_pre_samples)
    if single_run:
        print("Setup took %.2f s" % (time.time() - setup_start))

    teacher.enroll_learner(learner)

    return teacher


def setup_learning(args, number_range, single_run):
    concept = LetterAddition(args.problem_len)

    prior_distribution = np.ones(len(concept.get_concept_space()))
    prior_distribution /= np.sum(prior_distribution)

    if not args.manual:
        learner = create_simulated_learner(args, concept, number_range, prior_distribution)
    else:
        learner = HumanLearner(concept)

    belief = create_belief_model(args, prior_distribution, concept)
    teacher = create_teacher(args, concept, belief, learner, single_run)

    return concept, prior_distribution, learner, belief, teacher


def handle_single_run_end(global_time_start, learner, success, teacher):
    if not success:
        teacher.reveal_answer()
        print("# Concept not learned in expected time")
        print("Last guess:")
        print(learner.concept_belief)

    print("Learning time taken: %.2f" % learner.total_time)
    print("Global time elapsed: %.2f" % (time.time() - global_time_start))
    # print(teacher.action_history)

    plot_single_errors(teacher.assessment_history)
    plot_single_actions(teacher.action_history)


def handle_multi_run_end(action_history, error_history, global_time_start, time_history):
    plot_multi_errors(error_history)
    plt.clf()

    plot_multi_time(time_history)
    plt.clf()

    plot_multi_actions(action_history)
    plt.clf()

    print_statistics_table(error_history, time_history)
    print("Global time elapsed: %.2f" % (time.time() - global_time_start))


def describe_arguments(args):
    if args.manual:
        print("Using manual mode")
    else:
        if args.sim_model == 'memoryless':
            if args.sim_model_mode == 'pair':
                print("Using a simulated memoryless learner with pairwise updating")
            else:
                print("Using a simulated memoryless learner with stochastic updating")
        elif args.sim_model == 'discrete':
            if args.sim_model_mode == 'pair':
                print("Using a simulated learner with discrete memory (s=%d) and pairwise updating"
                      % args.sim_discrete_memory)
            else:
                print("Using a simulated learner with discrete memory (s=%d) and stochastic updating"
                      % args.sim_discrete_memory)
        elif args.sim_model == 'continuous':
            print("Using a simulated learner with continuous memory")
        if args.sim_no_noise:
            print("-- ignoring noise for simulated learners")

    if args.policy == 'random':
        print("Choosing actions randomly")
    else:
        if args.planning_model == 'memoryless':
            print("Planning actions using a memoryless belief model")
        elif args.planning_model == 'discrete':
            print("Planning actions using a belief model with discrete memory (s=%d)" % args.plan_discrete_memory)
        elif args.planning_model == 'continuous':
            print("Planning actions using a belief model with continuous memory")
        if args.plan_no_noise:
            print("-- ignoring noise in belief updating")



def main():
    parser = setup_arguments()
    args = parser.parse_args()
    describe_arguments(args)

    single_run = args.single_run or args.sim_count == 1
    number_range = list(range(0, args.number_range))

    global_time_start = time.time()

    error_history = []
    action_history = []
    time_history = []

    for i in range(args.sim_count):
        random.seed(args.sim_seed + i)
        np.random.seed(args.sim_seed + i)

        concept, prior_distribution, learner, belief, teacher = setup_learning(args, number_range, single_run)

        success = teacher.teach()

        if single_run:
            handle_single_run_end(global_time_start, learner, success, teacher)

            break
        else:
            if not success:
                print("x", end="")
            else:
                print(".", end="")

        action_history.append(teacher.action_history)
        error_history.append(teacher.assessment_history)
        time_history.append(learner.total_time)

    if not single_run:
        handle_multi_run_end(action_history, error_history, global_time_start, time_history)


main()
