# import numpy as np
import random

random.seed()


# problem: alphabetic arithmetic
class AlphaArithmetic:
    OP_PLUS = '+'
    OP_MINUS = '-'
    OP_MUL = '*'
    OP_DIV = '/'

    OPERATIONS = {
        OP_PLUS: lambda x, y: x + y,
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


class Teacher:
    def __init__(self, learning_phase_len, max_actions, problem_len):
        self.learning_phase_len = learning_phase_len
        self.max_actions = max_actions

        self.create_problem(problem_len)

    def create_problem(self, problem_len):
        elements = {}
        start = ord('A')
        # TODO start at 0 or 1?
        numbers = list(range(1, problem_len + 1))
        for i in range(problem_len):
            letter = random.sample(numbers, 1)[0]
            elements[chr(start + i)] = letter
            numbers.remove(letter)

        self.concept = AlphaArithmetic(elements)

    def teach(self):
        shown_concepts = []

        for action_num in range(self.max_actions):
            result, output = self.choose_action(shown_concepts)

            print(output)

            input("Continue?")

            if (action_num + 1) % self.learning_phase_len == 0:
                shown_concepts = []
                if self.assess():
                    return True

        return False

    def choose_action(self, shown_concepts):
        # only example action right now
        result, output = self.generate_example()
        while result in shown_concepts:
            result, output = self.generate_example()

        shown_concepts.append(result)

        return result, output

    def generate_example(self):
        equation, result = self.concept.generate_equation(2)

        return [equation, result], " ".join(equation) + " = " + str(result)

    def generate_question_with_feedback(self):

        return None

    def generate_quiz(self):

        return None

    def assess(self):
        # assessment time
        print("Do you know the answers?")
        guesses = []
        correct = True
        for item in self.concept.item_values:
            curr_guess = input("What is %s?" % item)
            if int(curr_guess) != self.concept.item_values[item]:
                correct = False
            guesses.append(curr_guess)

        if correct:
            print("Great! You learned all characters correctly")
            return True
        else:
            print("Nice try but there are some errors. Let's review some more...")
            return False

    def reveal_answer(self):
        print(self.concept.item_values)


teacher = Teacher(3, 40, 6)

if not teacher.teach():
    teacher.reveal_answer()

# teacher
# - has learner model
# - choose action: example, quiz, question with feedback
# - evaluate response
# - phases: 3 actions followed by assessment
# - estimates believed concepts distribution


# learner
# - real with interface
# - simulated
