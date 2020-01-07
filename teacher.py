import numpy as np
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

    def __init__(self, concept: ConceptBase, learning_phase_len: int = 3, max_phases: int = 40):
        self.learning_phase_len = learning_phase_len
        self.max_phases = max_phases

        self.gamma = 0.99

        # for memoryless model
        self.transition_noise = 0.15
        self.production_noise = 0.019

        self.concept = concept

        self.actions = {Actions.EXAMPLE: self.concept.generate_example,
                        Actions.QUIZ: self.concept.generate_quiz,
                        Actions.QUESTION: self.concept.generate_question_with_feedback}

        self.concept_space = self.concept.get_concept_space()

        self.strategy = self.choose_random_action
        # self.strategy = self.choose_best

        self.concept_space_size = 0
        self.prior_distribution = None
        self.belief = None
        self.true_concept_pos = -1
        self.best_action_stack = []
        self.learner = None

        self.action_history = []
        self.assessment_history = []
        self.verbose = False

    def setup(self, preplan_len: int = 9):
        # uniform prior distribution
        self.concept_space_size = len(self.concept_space)
        self.prior_distribution = np.array([1 / self.concept_space_size for _ in range(self.concept_space_size)])

        self.belief = Belief(self.prior_distribution.copy(), self.prior_distribution, self.concept,
                             verbose=self.verbose)

        # position of true concept
        # self.true_concept_pos = np.argmax(self.concept_space == self.concept.get_true_concepts())
        self.true_concept_pos = -1
        true_concept = self.concept.get_true_concepts()
        for i in range(len(self.concept_space)):
            if np.all(self.concept_space[i] == true_concept):
                self.true_concept_pos = i
                break

        self.best_action_stack = self.plan_best_actions(preplan_len, [10]*preplan_len)
        #print(self.best_action_stack)

    def teach(self):
        shown_concepts = []

        for action_num in range(self.max_phases*3):
            action_type, equation, result = self.choose_action(shown_concepts)

            action_data = (equation, result)

            if action_type == Actions.EXAMPLE:
                if self.verbose:
                    print("Let's see an example")
                self.learner.see_example(action_data)
                response = None
            elif action_type == Actions.QUIZ:
                if self.verbose:
                    print("Can you answer this quiz?")
                response = self.learner.see_quiz(action_data)
            else:
                # Question with feedback
                if self.verbose:
                    print("Question:")
                response = self.learner.see_question(action_data)

                correct = response == action_data[1]

                self.learner.see_question_feedback(action_data, correct)

            self.belief.update_belief(action_type, action_data, response)

            #print("-- Current likely concepts: %d" % np.count_nonzero(
            #    self.belief.belief_state > np.min(self.belief.belief_state)))

            #print("-- Contains correct concept?",
            #    self.belief.belief_state[self.true_concept_pos] > np.min(self.belief.belief_state))

            self.learner.finish_action()

            self.action_history.append((action_type, action_data))

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
            return self.plan_best_actions(2, [7, 6]).pop(0)

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

    def plan_best_actions(self, count, samples):
        # TODO use faster array/list method
        tree = {
            "belief": self.belief,
            "children": []
        }
        self.forward_plan(tree, count, samples)

        #self.print_plan_tree(tree)

        # find optimal path
        actions = self.find_optimal_action_path(tree)

        return actions

    def find_optimal_action_path(self, tree):
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
            print("%s Action: %s %s : %.2f" % (indent, item["action"], self.concept.gen_readable_format(item["item"], True), item["value"]))
            self.print_plan_tree(item, indent+'..')

    def forward_plan(self, parent, depth, sample_lens: list = None):

        if depth <= 0:
            # estimate value of leaf: based on the estimated probability that the student knows the correct concept
            return self.estimate_belief(parent["belief"])

        # ## simulate actions
        # if action of type example or question with feedback is chosen, the state of the learner is expected
        # to transition to a state consistent with it; thus the new belief state is the uniform distribution
        # over all concepts consistent with the sample

        combinations = self.concept.get_rl_actions()

        if sample_lens:
            sample_indices = random.sample(list(range(len(combinations))), k=sample_lens[0])

            samples = [combinations[i] for i in sample_indices]
        else:
            samples = combinations

        # check all options until a certain depth

        #print("Checking depth %d with %d options" % (depth, len(samples)))

        min_costs = float('Inf')

        for pair in samples:
            equation = pair[0]
            teaching_action = pair[1]
            value = self.concept.evaluate_concept([equation])
            result = (equation, value)

            # estimate observation
            expected_obs = None
            if teaching_action != Actions.EXAMPLE:
                # TODO is that correct?
                # sample observations?
                # believed_concept = random.choices(self.concept_space, weights=parent["belief"].belief_state)
                believed_concept_id = np.random.choice(self.concept_space_size, p=parent["belief"].belief_state)
                expected_obs = self.concept.evaluate_concept([equation], self.concept_space[believed_concept_id])

            # simulate belief change
            new_belief = Belief(parent["belief"].belief_state.copy(), self.prior_distribution, self.concept,
                                verbose=self.verbose)

            if teaching_action != Actions.QUIZ:
                new_belief.update_belief(teaching_action, result, expected_obs)

            new_node = {
                "belief": new_belief,
                "children": [],
                "item": result,
                "action": teaching_action
            }

            # approximate expected state & belief
            # go deeper
            val = self.forward_plan(new_node, depth-1, sample_lens[1:] if sample_lens else None)
            val = val * self.gamma + self.ACTION_COSTS[teaching_action]
            # propagate back up
            new_node["value"] = val

            parent["children"].append(new_node)

            if val < min_costs:
                min_costs = val

        parent["min_val"] = min_costs

        # if self.verbose and depth > 1:
        #     print("Finished level %d" % depth)

        return min_costs

    def estimate_belief(self, belief: Belief):
        # cost for a leaf node to be the probability of not passing the assessment phase multiplied by 10 * min_a(r(a))
        return (1 - belief.belief_state[self.true_concept_pos]) * 10 * min(self.ACTION_COSTS.values())

    def enroll_learner(self, learner):
        self.learner = learner
