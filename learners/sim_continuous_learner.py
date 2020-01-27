import itertools
import time

import numpy as np

from collections import deque

from concepts.concept_base import ConceptBase
from learners.base_learner import BaseLearner
from learners.sim_memoryless_learner import SimMemorylessLearner


class SimContinuousLearner(BaseLearner):
    def __init__(self, concept: ConceptBase, number_range: list, prior_distribution):
        super().__init__(concept)

        self.verbose = True
        self.pause = 0

        self.number_range = number_range

        self.prior_distribution = prior_distribution

        self.total_time = 0

        self.example_time = 7.0
        self.quiz_time = 6.6
        self.question_time = 12.0

        self.transition_noise = 0.14  # pretty high
        self.production_noise = 0.12

        # distribution about possibilities
        self.concept_belief = prior_distribution.copy()
        self.concepts = self.concept.get_concept_space()

        self.assessment_guess = None

        self.concept_action_values = {}
        self.pre_calc_state_values()

    def pre_calc_state_values(self):
        for action in self.concept.get_rl_actions():
            self.concept_action_values[action] = np.zeros(len(self.concepts))
            for idx, state in enumerate(self.concepts):
                self.concept_action_values[action][idx] = self.concept.evaluate_concept((action,), state, idx)

    def update_state(self, example):
        if np.random.random() < self.transition_noise:
            # ignore change
            return

        inconsistent_concepts = self.concept_action_values[example[0]] != example[1]
        self.concept_belief[inconsistent_concepts] = 0

        self.concept_belief /= np.sum(self.concept_belief)

    def see_example(self, example):
        self.assessment_guess = None

        self.print(self.concept.gen_readable_format(example))
        time.sleep(self.pause)

        self.update_state(example)

        self.total_time += self.example_time

    def see_quiz(self, quiz):
        self.assessment_guess = None

        answer_sample = self.generate_answer(quiz)

        self.total_time += self.quiz_time

        return answer_sample

    def generate_answer(self, quiz):
        self.print(self.concept.gen_readable_format(quiz, False))
        time.sleep(self.pause)

        answers = {}
        for result in self.concept.get_observation_space():
            concepts_w_result = self.concept_action_values[quiz[0]] == result
            answers[result] = np.sum(self.concept_belief[concepts_w_result])

        answer_sample = np.random.choice(list(answers.keys()), p=list(answers.values()))

        if np.random.random() <= self.production_noise:
            answer_sample -= 1  # TODO how do you define "inconsistent" for a probabilistic answer?

        self.print("I think it is %d" % answer_sample)

        return answer_sample

    def see_question_question(self, question):
        self.assessment_guess = None

        return self.generate_answer(question)

    def see_question_feedback(self, question, correct):
        if not correct:
            self.print("Not quite, the correct answer is %d" % question[1])
        else:
            self.print("Correct")

        time.sleep(self.pause)
        self.update_state(question)

        self.total_time += self.question_time

    def answer(self, item):
        if self.assessment_guess is None:
            concept_id = np.random.choice(len(self.concept_belief), p=self.concept_belief)
            self.assessment_guess = self.concepts[concept_id]

        curr_guess = self.assessment_guess[item[0]]

        self.print("I think %s is %d" % (item[1], curr_guess))

        return curr_guess

    def print(self, message):
        if self.verbose:
            print(message)
