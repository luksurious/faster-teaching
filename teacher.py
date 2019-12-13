import numpy as np
import itertools
import random

from belief import Belief
from concepts.concept_base import ConceptBase

from actions import Actions


class Teacher:
    ACTION_COSTS = {
        Actions.EXAMPLE: 7.0,
        Actions.QUIZ: 6.6,
        Actions.QUESTION: 12.0
    }

    def __init__(self, concept: ConceptBase, learning_phase_len: int = 3, max_actions: int = 40):
        self.learning_phase_len = learning_phase_len
        self.max_actions = max_actions

        self.gamma = 0.99

        self.strategy = self.choose_best

        # for memoryless model
        self.transition_noise = 0.15
        self.production_noise = 0.019

        self.concept = concept

        self.actions = {Actions.EXAMPLE: self.concept.generate_example,
                        Actions.QUIZ: self.concept.generate_quiz,
                        Actions.QUESTION: self.concept.generate_question_with_feedback}

        self.concept_space = self.concept.get_concept_space()

        # uniform prior distribution
        self.concept_space_size = len(self.concept_space)
        self.prior_distribution = np.array([1 / self.concept_space_size for _ in range(self.concept_space_size)])
        # self.belief_state = self.prior_distribution.copy()

        self.belief = Belief(self.prior_distribution.copy(), self.prior_distribution, self.concept)

        # position of true concept
        self.true_concept_pos = np.argmax(self.concept_space == self.concept.get_true_concepts())

        self.best_action_stack = self.plan_best_actions(3)
        print(self.best_action_stack)

    def teach(self):
        shown_concepts = []

        for action_num in range(self.max_actions):
            action_type, equation, result = self.choose_action(shown_concepts)

            action_data = (equation, result)

            if action_type == Actions.EXAMPLE:
                print("Let's see an example")
                response = None
                print(self.concept.gen_readable_format(action_data))
            elif action_type == Actions.QUIZ:
                print("Can you answer this quiz?")
                response = input(self.concept.gen_readable_format(action_data, False))  # TODO handle string input

                correct = response == action_data[1]
            else:
                # Question with feedback
                print("Question:")
                response = input(self.concept.gen_readable_format(action_data))  # TODO handle string input

                correct = response == action_data[1]
                if correct:
                    print("Yes, that's correct")
                else:
                    print("Not quite, the correct answer is %d" % action_data[1])
            self.belief.update_belief(action_type, action_data, response)

            print("Current likely concepts: %d" % np.count_nonzero(
                self.belief.belief_state > np.min(self.belief.belief_state)))

            print("Contains correct concept?",
                  self.belief.belief_state[self.true_concept_pos] > np.min(self.belief.belief_state))

            input("Continue?")

            if (action_num + 1) % self.learning_phase_len == 0:
                shown_concepts = []
                if self.assess():
                    return True

        return False

    def choose_action(self, shown_concepts):
        return self.strategy(shown_concepts)

    def choose_best(self, shown_concepts):
        if len(self.best_action_stack) > 0:
            # use precomputed actions
            return self.best_action_stack.pop(0)
        else:
            return self.plan_best_actions(2).pop(0)

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
        print("Do you know the answers?")
        correct = self.concept.assess()

        if correct:
            print("Great! You learned all characters correctly")
            return True
        else:
            print("Nice try but there are some errors. Let's review some more...")
            return False

    def reveal_answer(self):
        print(self.concept.get_true_concepts())

    def plan_best_actions(self, count):
        # TODO use faster array/list method
        tree = {
            "belief": self.belief,
            "children": []
        }
        self.forward_plan(tree, count, 10)

        #self.print_plan_tree(tree)

        # find optimal path
        actions = []
        parent = tree
        while len(parent["children"]) > 0:
            candidates = [el for el in parent["children"] if np.allclose([el["value"]], [parent["min_val"]])]

            # next_action_tree = parent["children"][ parent["min_idx"] ]
            next_action_tree = np.random.choice(candidates, 1)[0]

            actions.append((next_action_tree["action"], next_action_tree["item"][0], next_action_tree["item"][1]))
            parent = next_action_tree

        return actions

    def print_plan_tree(self, parent, indent=''):
        for item in parent["children"]:
            print("%s Action: %s : %.2f" % (indent, self.concept.gen_readable_format(item["item"], True), item["value"]))
            self.print_plan_tree(item, indent+'..')

    def forward_plan(self, parent, depth, sample_actions):

        if depth <= 0:
            # estimate value of leaf: based on the estimated probability that the student knows the correct concept
            return self.estimate_belief(parent["belief"]), 0

        # possible_paths = []

        # for action_type, action in self.actions.items():
        # ## simulate actions
        # if action of type example or question with feedback is chosen, the state of the learner is expected
        # to transition to a state consistent with it; thus the new belief state is the uniform distribution
        # over all concepts consistent with the sample

        # TODO since concept size is larger than equation size, should all combinations be evaluated or
        #  just one? So for the same concepts there are many different combinations possible

        # test all options
        concept = self.concept.get_true_concepts()
        # TODO: conversion to list only to count len later
        combinations = list(itertools.combinations(concept, 2))

        # TODO in the paper it is always talked about sampling actions,
        #  but in the figure it samples items, and considers all actions, and it also makes more sense?
        # samples = np.random.choice(self.concept_space, p=self.belief_state, size=sample_actions)
        # check all options until a certain depth
        samples = combinations
        #print("Checking depth %d with %d options" % (depth, len(samples)))

        min_costs = float('Inf')
        min_idx = -1

        for pair in samples:
            equation = [pair[0], '+', pair[1]]
            value = self.concept.evaluate_concept([equation], concept)
            result = [equation, value]

            # sample observations?

            # example action
            # # simulate belief change
            # estimate observation
            new_belief = Belief(parent["belief"].belief_state.copy(), self.prior_distribution, self.concept)
            new_belief.update_belief(Actions.EXAMPLE, result, None)

            new_node = {
                "belief": new_belief,
                "children": [],
                "item": result,
                "action": Actions.EXAMPLE
            }

            # approximate expected state & belief
            # go deeper
            val, idx = self.forward_plan(new_node, depth-1, sample_actions)
            val = val * self.gamma + self.ACTION_COSTS[Actions.EXAMPLE]
            # propagate back up
            new_node["value"] = val
            new_node["min_idx"] = idx

            parent["children"].append(new_node)

            if val < min_costs:
                min_costs = val
                min_idx = len(parent["children"])-1

            # quiz action

            # question action

        parent["min_val"] = min_costs
        parent["min_idx"] = min_idx

        return min_costs, min_idx

    def estimate_belief(self, belief: Belief):
        # cost for a leaf node to be the probability of not passing the assessment phase multiplied by 10 * min_a(r(a))
        return (1 - belief.belief_state[self.true_concept_pos]) * 10 * min(self.ACTION_COSTS.values())
