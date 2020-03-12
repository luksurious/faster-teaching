import random
import time

from copy import deepcopy

import numpy as np

from actions import Actions, ACTION_COSTS
from concepts.concept_base import ConceptBase
from learner_models.base_belief import BaseBelief
from planners.base_planner import BasePlanner


class ForwardSearchPlanner(BasePlanner):
    def __init__(self, concept: ConceptBase, actions: list, belief: BaseBelief,
                 plan_samples=None, plan_horizon: int = 2, verbose: bool = False):
        super().__init__(concept, actions)

        self.action_count = 0

        self.gamma = 0.99

        self.best_action_stack = {
            'action': None,
            'responses': {}
        }

        if plan_samples is None:
            plan_samples = [5, 5]
        self.plan_samples = plan_samples
        self.plan_horizon = plan_horizon

        self.belief = belief

        self.concept_space = self.concept.get_concept_space()

        self.verbose = verbose

        # position of true concept
        self.true_concept_pos = -1
        true_concept = self.concept.get_true_concepts()
        for i in range(len(self.concept_space)):
            if np.all(self.concept_space[i] == true_concept):
                self.true_concept_pos = i
                break

    def perform_preplanning(self, preplan_len: int = 9, preplan_samples: int = 10):
        self.preplan_walker(self.best_action_stack, preplan_len, self.belief.copy(),
                            [preplan_samples] * self.plan_horizon)

        return self.best_action_stack

    def preplan_walker(self, parent, level, belief, samples):
        # type, item, value
        next_action = self.plan_best_action(self.plan_horizon, samples, belief)

        parent['action'] = next_action
        if level == 1:
            return

        result = (next_action[1], next_action[2])
        if next_action[0] == Actions.QUIZ:
            result = (next_action[1], None)

        if next_action[0] == Actions.EXAMPLE:
            responses = [None]
        else:
            responses = self.concept.get_observation_space()

        for response in responses:
            parent['responses'][response] = {
                'action': None,
                'responses': {}
            }
            new_belief = belief.copy()
            new_belief.update_belief(next_action[0], result, response)

            self.preplan_walker(parent['responses'][response], level - 1, new_belief, samples)

    def load_preplanning(self, data):
        self.best_action_stack = deepcopy(data)

    def choose_action(self, prev_response=None):
        self.action_count += 1

        if self.action_count > 1 and self.best_action_stack.get('responses'):
            self.best_action_stack = self.best_action_stack['responses'].get(prev_response)

        if self.best_action_stack and self.best_action_stack.get('action'):
            # use precomputed actions
            return self.best_action_stack['action']
        else:
            return self.plan_best_action(self.plan_horizon, self.plan_samples)

    def start_teaching_phase(self):
        pass

    def reset(self):
        self.action_count = 0

    def plan_best_action(self, horizon: int, samples: list, belief: BaseBelief = None):
        if len(samples) < horizon:
            samples = [samples[0]] * horizon

        if belief is None:
            belief = self.belief

        start_time = time.time()

        tree = {
            "children": []
        }
        self.forward_plan(belief.copy(), tree, horizon, samples)

        if self.verbose:
            print("// planning took %.2f" % (time.time() - start_time))

        # self.print_plan_tree(tree)

        # find optimal path
        action = self.find_optimal_action_path(tree)

        return action

    @staticmethod
    def find_optimal_action_path(tree, depth=1):
        actions = []
        parent = tree
        while len(parent["children"]) > 0 and depth > 0:
            min_indices = np.flatnonzero(parent["costs"] == parent["costs"].min())
            candidates = [parent["children"][i] for i in min_indices]

            next_action_tree = np.random.choice(candidates, 1)[0]

            actions.append((next_action_tree["action"], next_action_tree["item"][0], next_action_tree["item"][1]))
            parent = next_action_tree
            depth -= 1

        if len(actions) == 1:
            return actions[0]

        return actions

    def print_plan_tree(self, parent, indent=''):
        for item in parent["children"]:
            print("%s Action: %s %s : %.2f" % (
                indent, item["action"], self.concept.gen_readable_format(item["item"], True), item["value"]))
            self.print_plan_tree(item, indent + '..')

    def forward_plan(self, belief: BaseBelief, parent, depth, sample_lens: list = None):

        if depth <= 0:
            # estimate value of leaf: based on the estimated probability that the student knows the correct concept
            return self.estimate_belief(belief)

        samples = self.sample_planning_items(sample_lens)

        child_sample_len = sample_lens[1:] if sample_lens else None

        # save state to reset to later
        model_state = belief.get_state()

        parent["costs"] = np.zeros(len(samples) * len(self.actions))
        item_index = 0

        min_cost = float("Inf")

        for item in samples:
            value = self.concept.evaluate_concept([item])
            result = (item, value)

            for teaching_action in self.actions:

                val = ACTION_COSTS[teaching_action]

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
        return (1 - belief.get_concept_prob(self.true_concept_pos)) * 10 * min(ACTION_COSTS.values())
