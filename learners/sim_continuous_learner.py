import time

import numpy as np

from actions import Actions
from concepts.concept_base import ConceptBase
from learners.base_learner import BaseLearner
from random_ng import rand_ng


class SimContinuousLearner(BaseLearner):
    def __init__(self, concept: ConceptBase, prior_distribution: np.ndarray):
        super().__init__(concept)

        self.verbose = True
        self.pause = 0

        self.prior_distribution = prior_distribution

        self.total_time = 0

        self.transition_noise = concept.TRANS_NOISE['continuous']
        self.production_noise = concept.PROD_NOISE['continuous']

        # distribution about possibilities
        self.concept_belief = prior_distribution.copy()
        self.concepts = self.concept.get_concept_space()

        self.assessment_guess = None

        self.concept_action_values = self.concept.state_action_values

    def update_state(self, example):
        if rand_ng.rg.random() < self.transition_noise:
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

        self.total_time += self.concept.ACTION_COSTS[Actions.EXAMPLE]

    def see_quiz(self, quiz):
        self.assessment_guess = None

        answer_sample = self.generate_answer(quiz)

        self.total_time += self.concept.ACTION_COSTS[Actions.QUIZ]

        return answer_sample

    def generate_answer(self, quiz):
        self.print(self.concept.gen_readable_format(quiz, False))
        time.sleep(self.pause)

        if rand_ng.rg.random() <= self.production_noise:
            # produce random answer
            answer_sample = rand_ng.rg.choice(self.concept.get_observation_space())
        else:
            answers = {}
            for result in self.concept.get_observation_space():
                concepts_w_result = self.concept_action_values[quiz[0]] == result
                answers[result] = np.sum(self.concept_belief[concepts_w_result])

            answer_sample = rand_ng.rg.choice(list(answers.keys()), p=list(answers.values()))

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

        self.total_time += self.concept.ACTION_COSTS[Actions.FEEDBACK]

    def answer(self, item):
        if self.assessment_guess is None:
            concept_id = rand_ng.rg.choice(len(self.concept_belief), p=self.concept_belief)
            self.assessment_guess = self.concepts[concept_id]

        curr_guess = self.concept.evaluate_concept(item[0], self.assessment_guess)

        self.print("I think %s is %d" % (item[1], curr_guess))

        return curr_guess

    def print(self, message):
        if self.verbose:
            print(message)
