import time

import numpy as np

from actions import Actions
from concepts.concept_base import ConceptBase
from concepts.letter_addition import LetterAddition
from random_ng import rand_ng
from .base_learner import BaseLearner


class SimMemorylessLearner(BaseLearner):
    def __init__(self, concept: ConceptBase, prior_distribution: np.ndarray):
        super().__init__(concept)

        self.verbose = True
        self.pause = 0

        self.transition_noise = concept.TRANS_NOISE['memoryless']
        self.production_noise = concept.PROD_NOISE['memoryless']

        self.number_range = None
        if isinstance(concept, LetterAddition):
            self.number_range = concept.numbers

        # what should it be initialized?
        self.concept_space = concept.get_concept_space()
        concept_space_len = len(self.concept_space)
        self.prior_distribution = prior_distribution

        self.concept_belief = self.concept_space[rand_ng.rg.choice(range(concept_space_len),
                                                                                p=self.prior_distribution)]

        self.total_time = 0

        self.mode = "stoch"

    def see_example(self, example):
        self.print(self.concept.gen_readable_format(example))
        time.sleep(self.pause)

        believed_answer = self.self_evaluate(example[0])
        if believed_answer == example[1]:
            # current concept consistent with example
            pass
        else:
            self.update_state(example)

        self.total_time += self.concept.ACTION_COSTS[Actions.EXAMPLE]

    def see_quiz(self, quiz):
        response = self.generate_answer(quiz)

        self.total_time += self.concept.ACTION_COSTS[Actions.QUIZ]

        return response

    def generate_answer(self, quiz):
        self.print(self.concept.gen_readable_format(quiz, False))
        time.sleep(self.pause)

        if rand_ng.rg.random() < self.production_noise:
            response = rand_ng.rg.choice(list(self.concept.get_observation_space()))  # random answer
        else:
            response = self.self_evaluate(quiz[0])

        self.print("I think it is %d" % response)

        return response

    def see_question_question(self, question):
        return self.generate_answer(question)

    def see_question_feedback(self, question, correct):
        if not correct:
            self.print("Not quite, the correct answer is %d" % question[1])

            self.update_state(question)
        else:
            self.print("Correct")

        self.total_time += self.concept.ACTION_COSTS[Actions.FEEDBACK]
        time.sleep(self.pause)

    def update_state(self, example):
        if rand_ng.rg.random() < self.transition_noise:
            # ignore change
            return

        if self.mode == "pair":
            # TODO improve: properly calculate state distances
            possible_pairs = self.generate_possible_pairs(example[1])

            # TODO improve: prefer options with a match of current belief? i.e. least changes
            pair = rand_ng.rg.choice(possible_pairs)

            self.update_values_with_pair(example[0], pair)

            self.fill_empty_mappings()
        else:
            # Sample concept consistent with action according to prior
            concepts_results = np.array([self.concept.evaluate_concept(example[0], c) for c in self.concept_space])
            consistent_concepts_filter = concepts_results == example[1]

            consistent_concepts = np.flatnonzero(consistent_concepts_filter)

            consistent_concepts_prob = self.prior_distribution[consistent_concepts_filter]
            consistent_concepts_prob /= np.sum(consistent_concepts_prob)

            new_belief_idx = rand_ng.rg.choice(consistent_concepts, p=consistent_concepts_prob)
            self.concept_belief = self.concept_space[new_belief_idx]

    def update_values_with_pair(self, letters, pair):
        # mark values from the pick as invalid
        for idx, val in enumerate(self.concept_belief):
            if val == pair[0] or val == pair[1]:
                self.concept_belief[idx] = -1

        # set new values from the picked pair
        self.concept_belief[letters[0]] = pair[0]
        self.concept_belief[letters[1]] = pair[1]

    def generate_possible_pairs(self, result):
        possible_pairs = []
        for i in range(int(result) + 1):
            pair = (i, int(result) - i)
            if max(pair[0], pair[1]) > max(self.number_range):
                continue

            if pair[0] == pair[1]:
                continue

            possible_pairs.append(pair)
        return possible_pairs

    def fill_empty_mappings(self):
        num_reassign, refill_idx = self.get_idx_val_to_fill()

        num_reassign = rand_ng.rg.choice(num_reassign, len(refill_idx), replace=False).tolist()
        for i in refill_idx:
            if self.concept_belief[i] == -1:
                self.concept_belief[i] = num_reassign.pop(0)

    def get_idx_val_to_fill(self):
        refill_idx = []
        num_reassign = np.array(self.number_range.copy())
        for i, val in enumerate(self.concept_belief):
            if val > -1:
                num_reassign[val] = -1
            else:
                refill_idx.append(i)

        num_reassign = num_reassign[num_reassign > -1]

        return num_reassign, refill_idx

    def self_evaluate(self, equation):
        return self.concept.evaluate_concept(equation, self.concept_belief)

    def answer(self, item):
        curr_guess = self.concept.evaluate_concept(item[0], self.concept_belief)
        curr_guess = self.concept.format_response(curr_guess)

        self.print("I think %s is %d" % (item[1], curr_guess))

        return curr_guess

    def print(self, message):
        if self.verbose:
            print(message)
