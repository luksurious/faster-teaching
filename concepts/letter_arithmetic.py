# TODO: use NP for random values?
# import numpy as np
import itertools
import random
from .concept_base import ConceptBase


# problem: alphabetic arithmetic
class LetterArithmetic(ConceptBase):
    OP_PLUS = '+'
    OP_MINUS = '-'
    OP_MUL = '*'
    OP_DIV = '/'

    OPERATIONS = {
        OP_PLUS: lambda x, y: x + y,

        # In the experiment described in the paper only additions were used
        # OP_MINUS: lambda x, y: x - y,
        # OP_MUL: lambda x, y: x * y,
        # OP_DIV: lambda x, y: x / y,
    }

    def __init__(self, problem_len: int):
        elements = {}
        start = ord('A')
        # TODO start at 0 or 1?
        self.numbers = list(range(0, problem_len + 1))
        self.letters = []

        assign_numbers = self.numbers.copy()
        for i in range(problem_len):
            number = random.sample(assign_numbers, 1)[0]
            elements[chr(start + i)] = number
            assign_numbers.remove(number)

            self.letters.append(chr(start + i))

        self.item_values = elements
        self.equation_length = 2

    def calc_pair(self, a: str, b: str, operation: str, values=None):
        if not values:
            values = self.item_values

        try:
            value_a = values[a]
            value_b = values[b]
        except KeyError as err_val:
            print("Invalid character given: %s" % err_val)
            return None

        try:
            return self.OPERATIONS[operation](value_a, value_b)
        except KeyError:
            print("Invalid operation given: %s" % operation)
            return None

    def generate_equation(self, length: int = 2):
        chars = random.sample(self.item_values.keys(), length)
        # if only using addition, order does not matter, so we can reduce the possibilities
        chars = sorted(chars)

        operations = random.sample(self.OPERATIONS.keys(), length - 1)

        equation = []
        for key in range(0, len(chars), 2):
            a = chars[key]
            b = chars[key + 1]
            op = operations.pop()
            equation.append(a)
            equation.append(op)
            equation.append(b)

        return equation

    def evaluation_equation(self, equation, values=None):
        result = 0
        start_letter = equation[0]
        for key in range(1, len(equation), 2):
            b = equation[key + 1]
            op = equation[key]

            result = self.calc_pair(start_letter, b, op, values)

            start_letter = result

        return result

    def generate_example(self, alternative_concept=None):
        equation = self.generate_equation(self.equation_length)
        result = self.evaluation_equation(equation, alternative_concept)

        return [equation, result], " ".join(equation) + " = " + str(result)

    def generate_question_with_feedback(self, alternative_concept=None):
        return self.generate_quiz(alternative_concept)

    def generate_quiz(self, alternative_concept=None):
        equation = self.generate_equation(self.equation_length)
        result = self.evaluation_equation(equation, alternative_concept)

        return [equation, result], " ".join(equation) + " = ?"

    def evaluate_concept(self, result, concept):
        concept_val = self.evaluation_equation(result[0], concept)

        return concept_val

    def assess(self) -> bool:
        guesses = []
        correct = True
        for item in self.item_values:
            curr_guess = input("What is %s?" % item)
            try:
                curr_guess = int(curr_guess)

                if curr_guess != self.item_values[item]:
                    correct = False
            except:
                correct = False
            guesses.append(curr_guess)

        return correct

    def get_true_concepts(self):
        return self.item_values

    def get_concept_space(self):
        all_number_combinations = list(itertools.permutations(self.numbers, len(self.item_values)))
        all_concepts = [{letter: comb[i] for i, letter in enumerate(self.letters)} for comb in all_number_combinations]

        return all_concepts
