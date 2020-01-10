# TODO: use NP for random values?
import numpy as np
import itertools
import random

from actions import Actions
from .concept_base import ConceptBase


# problem: alphabetic arithmetic
class LetterAddition(ConceptBase):
    def __init__(self, problem_len: int, number_range: list = None):
        elements = np.zeros(problem_len)
        start = ord('A')

        if number_range is None:
            # TODO start at 0 or 1?
            number_range = list(range(0, problem_len))

        self.numbers = number_range
        self.letters = []

        self.assign_numbers(elements, problem_len, start)

        self.item_values = elements
        self.equation_length = 2

        all_number_combinations = list(itertools.permutations(self.numbers, problem_len))
        # all_concepts = [{letter: comb[i] for i, letter in enumerate(self.letters)} for comb in all_number_combinations]
        self.all_concepts = np.array(all_number_combinations)

        self.letter_combs = list(itertools.combinations(range(problem_len), self.equation_length))
        # self.letter_actions = list(itertools.product(letter_combs, Actions.all()))

        self.possible_values = set([x[0] + x[1] for x in itertools.combinations(self.numbers, 2)])

    def assign_numbers(self, elements, problem_len, start):
        assign_numbers = self.numbers.copy()
        for i in range(problem_len):
            number = random.sample(assign_numbers, 1)[0]
            elements[i] = number
            assign_numbers.remove(number)

            self.letters.append(chr(start + i))

    # def calc_pair(self, a: str, b: str, values=None):
    #     if not values:
    #         values = self.item_values
    #
    #     try:
    #         value_a = values[a]
    #         value_b = values[b]
    #     except KeyError as err_val:
    #         print("Invalid character given: %s" % err_val)
    #         return None
    #
    #     return value_a + value_b

    def generate_equation(self, length: int = 2):
        chars = random.sample(range(len(self.letters)), length)
        # if only using addition, order does not matter, so we can reduce the possibilities
        chars = sorted(chars)

        return chars

    def evaluate_equation(self, equation, values=None):
        if values is None:
            values = self.item_values
        result = 0
        for letter_idx in equation:
            result += values[letter_idx]

        return result

    def generate_example(self, alternative_concept=None):
        equation = self.generate_equation(self.equation_length)
        result = self.evaluate_equation(equation, alternative_concept)

        return equation, result

    def generate_question_with_feedback(self, alternative_concept=None):
        return self.generate_quiz(alternative_concept)

    def generate_quiz(self, alternative_concept=None):
        equation = self.generate_equation(self.equation_length)
        result = self.evaluate_equation(equation, alternative_concept)

        return equation, result

    def gen_readable_format(self, result, show_answer=True):
        if show_answer is False or result[1] is None:
            right_side = '?'
        else:
            right_side = str(int(result[1]))

        letters = map(lambda i: self.letters[i], result[0])

        return " + ".join(letters) + " = " + right_side

    def evaluate_concept(self, result, concept=None):
        concept_val = self.evaluate_equation(result[0], concept)

        return int(concept_val)

    def assess(self, learner) -> (bool, float):
        guesses = []
        correct = True
        errors = 0
        for index, item in enumerate(self.letters):
            curr_guess = learner.answer((index, item))
            try:
                curr_guess = int(curr_guess)

                if curr_guess != self.item_values[index]:
                    correct = False
                    errors += 1
            except:
                correct = False
                errors += 1
            guesses.append(curr_guess)

        return correct, errors

    def get_true_concepts(self):
        return self.item_values

    def get_concept_space(self):
        return self.all_concepts

    def get_rl_actions(self, sample_count=None):
        return self.letter_combs

    def get_observation_space(self):
        return self.possible_values
