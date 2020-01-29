import time

import numpy as np
import random

from concepts.concept_base import ConceptBase

from actions import Actions
from learner_models.base_belief import BaseBelief


class Teacher:
    ACTION_COSTS = {
        Actions.EXAMPLE: 7.0,
        Actions.QUIZ: 6.6,
        Actions.QUESTION: 12.0
    }

    def __init__(self, concept: ConceptBase, belief: BaseBelief, is_random: bool, learning_phase_len: int = 3,
                 max_phases: int = 40):
        self.learning_phase_len = learning_phase_len
        self.max_phases = max_phases

        self.gamma = 0.99

        if is_random:
            self.strategy = self.choose_random_action
        else:
            self.strategy = self.choose_best

        self.true_concept_pos = -1
        self.best_action_stack = []
        self.learner = None

        self.action_history = []
        self.assessment_history = []
        self.verbose = False

        self.plan_samples = [5, 5]
        self.plan_horizon = 2

        self.concept = concept
        self.belief = belief

        self.actions = {Actions.EXAMPLE: self.concept.generate_example,
                        Actions.QUIZ: self.concept.generate_quiz,
                        Actions.QUESTION: self.concept.generate_question_with_feedback}

        self.concept_space = self.concept.get_concept_space()

        # TODO stop planning after max time
        self.max_time = 3

        self.action_count = 0

        # position of true concept
        self.true_concept_pos = -1
        true_concept = self.concept.get_true_concepts()
        for i in range(len(self.concept_space)):
            if np.all(self.concept_space[i] == true_concept):
                self.true_concept_pos = i
                break

    def setup(self, preplan_len: int = 9, preplan_samples: int = 10):
        self.best_action_stack = self.plan_best_actions(preplan_len, [preplan_samples]*preplan_len)

    def reset(self):
        self.action_history = []
        self.assessment_history = []
        self.action_count = 0
        self.belief.reset()

    def teach(self):
        shown_concepts = []

        for self.action_count in range(self.max_phases*3):
            action_type, equation, result = self.choose_action(shown_concepts)

            action_data = (equation, result)
            action_data_hidden = (equation, None)

            if action_type == Actions.EXAMPLE:
                if self.verbose:
                    print("Let's see an example")
                self.learner.see_example(action_data)
                response = None
            elif action_type == Actions.QUIZ:
                if self.verbose:
                    print("Can you answer this quiz?")
                response = self.learner.see_quiz(action_data_hidden)
                action_data = action_data_hidden
            else:
                # Question with feedback
                if self.verbose:
                    print("Question:")
                response = self.learner.see_question_question(action_data_hidden)

                correct = response == action_data[1]

                self.learner.see_question_feedback(action_data, correct)

            self.belief.update_belief(action_type, action_data, response)

            self.learner.finish_action(action_data)

            self.action_history.append((action_type, action_data))

            if (self.action_count + 1) % self.learning_phase_len == 0:
                shown_concepts = []
                if self.assess():
                    return True

        return False

    def choose_action(self, shown_concepts):
        return self.strategy(shown_concepts)

    def choose_best(self, shown_concepts):
        if self.action_count < len(self.best_action_stack):
            # use precomputed actions
            return self.best_action_stack[self.action_count]
        else:
            return self.plan_best_actions(self.plan_horizon, self.plan_samples).pop(0)

    def choose_random_action(self, shown_concepts):
        # random strategy
        current_type = random.sample(self.actions.keys(), 1)[0]

        equation, result = self.actions[current_type]()
        while equation in shown_concepts:
            equation, result = self.actions[current_type]()

        # TODO check for different types
        shown_concepts.append(equation)

        return current_type, equation, result

    def assess(self):
        # assessment time
        if self.verbose:
            print("Do you know the answers?")
        correct, errors = self.concept.assess(self.learner)

        self.assessment_history.append(errors)

        if correct:
            if self.verbose:
                print("Great! You learned all characters correctly")
            return True
        else:
            if self.verbose:
                print("Nice try but there are some errors. Let's review some more...")
            return False

    def reveal_answer(self):
        print("True answer:")
        print(self.concept.get_true_concepts())

    # TODO possibly extract
    def plan_best_actions(self, count: int, samples: list):
        if len(samples) < count:
            samples = [samples[0]]*count

        start_time = time.time()

        # TODO use faster array/list method
        tree = {
            "children": []
        }
        self.forward_plan(self.belief.copy(), tree, count, samples)

        if self.verbose:
            print("// planning took %.2f" % (time.time()-start_time))

        # self.print_plan_tree(tree)

        # find optimal path
        actions = self.find_optimal_action_path(tree)

        return actions

    @staticmethod
    def find_optimal_action_path(tree):
        actions = []
        parent = tree
        while len(parent["children"]) > 0:
            min_indices = np.flatnonzero(parent["costs"] == parent["costs"].min())
            candidates = [parent["children"][i] for i in min_indices]

            next_action_tree = np.random.choice(candidates, 1)[0]

            actions.append((next_action_tree["action"], next_action_tree["item"][0], next_action_tree["item"][1]))
            parent = next_action_tree

        return actions

    def print_plan_tree(self, parent, indent=''):
        for item in parent["children"]:
            print("%s Action: %s %s : %.2f" % (indent, item["action"], self.concept.gen_readable_format(item["item"], True), item["value"]))
            self.print_plan_tree(item, indent+'..')

    def forward_plan(self, belief: BaseBelief, parent, depth, sample_lens: list = None):

        if depth <= 0:
            # estimate value of leaf: based on the estimated probability that the student knows the correct concept
            return self.estimate_belief(belief)

        samples = self.sample_planning_items(sample_lens)

        child_sample_len = sample_lens[1:] if sample_lens else None

        # save state to reset to later
        model_state = belief.get_state()

        parent["costs"] = np.zeros(len(samples)*len(Actions.all()))
        item_index = 0

        min_cost = float("Inf")

        for item in samples:
            value = self.concept.evaluate_concept([item])
            result = (item, value)

            for teaching_action in Actions.all():

                val = self.ACTION_COSTS[teaching_action]

                new_node = {
                    "children": [],
                    "item": result,
                    "action": teaching_action
                }

                val += self.plan_single_action(belief, child_sample_len, depth, model_state, new_node, result,
                                               teaching_action,
                                               min_cost, val)

                # propagate back up
                new_node["value"] = val

                if val < min_cost:
                    min_cost = val

                parent["children"].append(new_node)

                parent["costs"][item_index] = val
                item_index += 1

        return parent["costs"].min()

    def plan_single_action(self, belief: BaseBelief, child_sample_len: list, depth: int, model_state, new_node, result,
                           teaching_action, best_val, action_cost):
        if teaching_action == Actions.EXAMPLE:
            # no observations
            expected_obs = None

            belief.update_belief(teaching_action, result, expected_obs)
            val = self.gamma * self.forward_plan(belief, new_node, depth - 1, child_sample_len)

            belief.set_state(model_state)
        else:
            val = 0

            if teaching_action == Actions.QUIZ:
                result = (result[0], None)  # No evidence is given to the learner

            for expected_obs in self.concept.get_observation_space():
                # TODO verify; could be precomputed
                obs_prob = belief.get_observation_prob(result, expected_obs)

                if obs_prob == 0:
                    continue

                belief.update_belief(teaching_action, result, expected_obs)

                val += self.gamma * obs_prob * self.forward_plan(belief, new_node, depth - 1, child_sample_len)

                belief.set_state(model_state)

                if (val + action_cost) > best_val:
                    # print("Canceled calculating more obs - cannot get better")
                    break

        return val

    def sample_planning_items(self, sample_lens):
        combinations = self.concept.get_rl_actions()
        if sample_lens:
            sample_indices = random.sample(list(range(len(combinations))), k=sample_lens[0])

            samples = [combinations[i] for i in sample_indices]
        else:
            samples = combinations

        return samples

    def estimate_belief(self, belief: BaseBelief):
        # TODO move to concept
        # cost for a leaf node to be the probability of not passing the assessment phase multiplied by 10 * min_a(r(a))
        return (1 - belief.get_concept_prob(self.true_concept_pos)) * 10 * min(self.ACTION_COSTS.values())

    def enroll_learner(self, learner):
        self.learner = learner
