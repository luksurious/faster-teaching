import math
import random
import time

import numpy as np

from concepts.concept_base import ConceptBase
from learners.base_learner import BaseLearner


PAUSE = 1


class SimLearner(BaseLearner):
    def __init__(self, concept: ConceptBase):
        super().__init__(concept)

        self.problem_len = len(concept.get_true_concepts())

        # what should it be initialized?
        self.letter_values = [i for i in range(self.problem_len)]

        # should also the memoryless learner have a distribution over concepts?

    def see_example(self, example):
        print(self.concept.gen_readable_format(example))
        time.sleep(PAUSE)

        believed_answer = self.self_evaluate(example[0])
        if believed_answer == example[1]:
            # current concept consistent with example
            pass
        else:
            self.update_state(example)

    def see_quiz(self, quiz):
        print(self.concept.gen_readable_format(quiz, False))
        time.sleep(PAUSE)
        response = self.self_evaluate(quiz[0])

        print("I think it is %d" % response)

        return response

    def see_question(self, question):
        print(self.concept.gen_readable_format(question, False))
        time.sleep(PAUSE)
        response = self.self_evaluate(question[0])
        print("I think it is %d" % response)

        return response

    def see_question_feedback(self, question, correct):
        time.sleep(PAUSE)
        if not correct:
            print("Not quite, the correct answer is %d" % question[1])
            self.update_state(question)

    def update_state(self, example):
        possible_pairs = []
        for i in range(int(example[1]) + 1):
            pair = (i, int(example[1]) - i)
            if pair[0] >= self.problem_len or pair[1] >= self.problem_len:
                continue

            if pair[0] == pair[1]:
                continue

            possible_pairs.append(pair)
        # pick randomly from possibilities?
        # prefer options with a match of current belief?
        pair = random.choice(possible_pairs)

        refill_idx = []
        for idx, val in enumerate(self.letter_values):
            if val == pair[0] or val == pair[1]:
                refill_idx.append(idx)
                self.letter_values[idx] = -1

        num_reassign = []
        if self.letter_values[example[0][0]] != -1:
            num_reassign.append(self.letter_values[example[0][0]])
        if self.letter_values[example[0][1]] != -1:
            num_reassign.append(self.letter_values[example[0][1]])

        self.letter_values[example[0][0]] = pair[0]
        self.letter_values[example[0][1]] = pair[1]

        for i in refill_idx:
            if self.letter_values[i] == -1:
                self.letter_values[i] = num_reassign.pop(0)

    def self_evaluate(self, equation):
        return self.letter_values[equation[0]] + self.letter_values[equation[1]]

    def answer(self, item):
        curr_guess = self.letter_values[item[0]]

        print("I think %s is %d" % (item[1], curr_guess))

        return curr_guess
