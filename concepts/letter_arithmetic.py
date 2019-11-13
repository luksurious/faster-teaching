# TODO: use NP for random values?
# import numpy as np
import random
from .concept_base import ConceptBase


# TODO where to seed
random.seed()


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

    def __init__(self, item_values: dict):
        self.item_values = item_values

    def calc_pair(self, a: str, b: str, operation: str):
        try:
            value_a = self.item_values[a]
            value_b = self.item_values[b]
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

        operations = random.sample(self.OPERATIONS.keys(), length - 1)

        result = 0
        equation = []
        for key in range(0, len(chars), 2):
            a = chars[key]
            b = chars[key + 1]
            op = operations.pop()
            equation.append(a)
            equation.append(op)
            equation.append(b)

            result += self.calc_pair(a, b, op)

        return equation, result

    def generate_example(self):
        equation, result = self.generate_equation(2)

        return [equation, result], " ".join(equation) + " = " + str(result)

    def generate_question_with_feedback(self):
        return self.generate_quiz()

    def generate_quiz(self):
        equation, result = self.generate_equation(2)

        return [equation, result], " ".join(equation) + " = ?"

    def assess(self) -> bool:
        guesses = []
        correct = True
        for item in self.item_values:
            curr_guess = input("What is %s?" % item)
            if curr_guess == '' or int(curr_guess) != self.item_values[item]:
                correct = False
            guesses.append(curr_guess)

        return correct

    def get_true_concepts(self):
        return self.item_values
