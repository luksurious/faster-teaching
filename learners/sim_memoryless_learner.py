import math
import random
import time

import numpy as np

from concepts.concept_base import ConceptBase
from learners.base_learner import BaseLearner


class SimMemorylessLearner(BaseLearner):
    def __init__(self, concept: ConceptBase, number_range: list, prior_distribution):
        super().__init__(concept)

        self.verbose = True
        self.pause = 0

        self.transition_noise = 0.15
        self.production_noise = 0.019

        self.number_range = number_range

        # what should it be initialized?
        # self.letter_values = [i for i in range(self.problem_len)]
        # np.random.shuffle(self.letter_values)
        self.concept_space = concept.get_concept_space()
        concept_space_len = len(self.concept_space)
        self.prior_distribution = prior_distribution

        self.concept_belief = self.concept_space[np.random.choice(range(concept_space_len), p=self.prior_distribution)]
        self.problem_len = len(self.concept_belief)

        self.total_time = 0

        self.example_time = 7.0
        self.quiz_time = 6.6
        self.question_time = 12.0

        self.mode = "pair"

    def see_example(self, example):
        self.print(self.concept.gen_readable_format(example))
        time.sleep(self.pause)

        believed_answer = self.self_evaluate(example[0])
        if believed_answer == example[1]:
            # current concept consistent with example
            pass
        else:
            self.update_state(example)

        self.total_time += self.example_time

    def see_quiz(self, quiz):
        response = self.generate_answer(quiz)

        self.total_time += self.quiz_time

        return response

    def generate_answer(self, quiz):
        self.print(self.concept.gen_readable_format(quiz, False))
        time.sleep(self.pause)

        response = self.self_evaluate(quiz[0])
        if np.random.random() < self.production_noise:
            response -= 1  # slips, TODO whats the proper way?
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

        self.total_time += self.question_time
        time.sleep(self.pause)

    def update_state(self, example):
        if np.random.random() < self.transition_noise:
            # ignore change
            return

        if self.mode == "pair":
            possible_pairs = self.generate_possible_pairs(example[1])
            # print(possible_pairs)

            # TODO: prefer options with a match of current belief? i.e. least changes
            pair = random.choice(possible_pairs)

            self.update_values_with_pair(example[0], pair)

            self.fill_empty_mappings()
            # assert unique
            assert len(set(self.concept_belief)) == self.problem_len, "Non-unique values assigned: %s" % self.concept_belief
        else:
            # Sample concept consistent with action according to prior
            concepts_results = np.array([self.concept.evaluate_concept(example, c) for c in self.concept_space])
            consistent_concepts_filter = concepts_results == example[1]

            consistent_concepts = self.concept_space[consistent_concepts_filter]

            consistent_concepts_prob = self.prior_distribution[consistent_concepts_filter]
            consistent_concepts_prob /= np.sum(consistent_concepts_prob)

            self.concept_belief = consistent_concepts[np.random.choice(range(len(consistent_concepts)),
                                                                       p=consistent_concepts_prob)]

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

        num_reassign = np.random.choice(num_reassign, len(refill_idx), replace=False).tolist()
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
        return self.concept.evaluate_concept([equation], self.concept_belief)

    def answer(self, item):
        curr_guess = self.concept_belief[item[0]]

        self.print("I think %s is %d" % (item[1], curr_guess))

        return curr_guess

    def print(self, message):
        if self.verbose:
            print(message)
