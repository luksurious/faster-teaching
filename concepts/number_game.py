import math
import random

import numpy as np

from actions import ACTION_COSTS_NUMBER
from concepts.concept_base import ConceptBase, ActionResult, ConceptItemBase


# 5 teaching actions per phase
# random: half inside, half outside concept
# assessment phase: show 5 random concepts from inside, 5 from outside, require correct answer to terminate
# precomputed: 20 actions
# continuous mode: H = 3, S=6+6+8, p=16 ; shouldnt it also make sure to check inside & outside samples for the planning?
# memoryless: H = 2, S=6+8
# discrete: H = 2, S=6+6, M=2
class NumberGame(ConceptBase):
    def __init__(self):
        super().__init__(ACTION_COSTS_NUMBER)
        self.range = range(1, 101)

        self.prior_lambda = 2/3

        self.inside_prob = 0.5

        self.concept_space, self.prior = self.generate_plausible_concepts()

        # self.cur_concept = NumberGameConcept(multiples=7, multiples_mod=-1)
        # self.cur_concept = NumberGameConcept(interval_start=64, interval_end=83)
        self.cur_concept = NumberGameConcept(multiples=7)

        self.true_concept_idx = self.find_true_concept_idx()

        self.numbers_inside, self.numbers_outside = [], []

        for i in self.range:
            if self.cur_concept.check(i):
                self.numbers_inside.append(i)
            else:
                self.numbers_outside.append(i)

    def find_true_concept_idx(self):
        for idx, concept in enumerate(self.concept_space):
            if concept == self.cur_concept:
                return idx

        return -1

    def generate_plausible_concepts(self):
        math_concepts, other_concepts = [], []
        math_concepts.append(NumberGameConcept(odd=True))
        math_concepts.append(NumberGameConcept(even=True))
        math_concepts.append(NumberGameConcept(square=True))
        math_concepts.append(NumberGameConcept(cube=True))
        math_concepts.append(NumberGameConcept(primes=True))

        for i in range(3, 13):
            math_concepts.append(NumberGameConcept(multiples=i))
            other_concepts.append(NumberGameConcept(multiples=i, multiples_mod=-1))
            other_concepts.append(NumberGameConcept(multiples=i, multiples_mod=1))

        for i in range(2, 11):
            math_concepts.append(NumberGameConcept(powers=i))

        for i in range(10):
            math_concepts.append(NumberGameConcept(ending=i))

        for a in range(1, 100):
            for b in range(a, 100):
                other_concepts.append(NumberGameConcept(interval_start=a, interval_end=b))

        math_count = len(math_concepts)
        math_prior = self.prior_lambda / math_count
        other_count = len(other_concepts)
        other_prior = (1-self.prior_lambda) / other_count

        concepts = math_concepts + other_concepts
        priors = [math_prior] * math_count + [other_prior] * other_count

        return concepts, np.array(priors)

    def get_default_prior(self) -> iter:
        return self.prior

    def assess(self, learner) -> (bool, float):
        guesses = []
        correct = True
        errors = 0

        cor_numbers = self.sample_inside(5)
        incor_numbers = self.sample_outside(5)

        questions = np.concatenate([cor_numbers, incor_numbers])

        np.random.shuffle(questions)

        for number, inside in questions:
            curr_guess = learner.answer((number, number))
            try:
                curr_guess = int(bool(curr_guess))

                if curr_guess != inside:
                    correct = False
                    errors += 1
            except:
                correct = False
                errors += 1
            guesses.append(curr_guess)

        return correct, errors

    def sample_inside(self, number: int):
        cor_numbers = np.random.choice(self.numbers_inside, number, replace=False)
        cor_numbers = np.stack([cor_numbers, [1] * number], axis=1)

        return cor_numbers

    def sample_outside(self, number: int):
        incor_numbers = np.random.choice(self.numbers_outside, number, replace=False)
        incor_numbers = np.stack([incor_numbers, [0] * number], axis=1)

        return incor_numbers

    def get_true_concept_idx(self):
        return self.true_concept_idx

    def get_concept_space(self) -> iter:
        return self.concept_space

    def generate_example(self) -> ActionResult:
        if np.random.random() > self.inside_prob:
            return np.random.choice(self.numbers_inside), True
        else:
            return np.random.choice(self.numbers_outside), False

    def generate_question_with_feedback(self) -> ActionResult:
        return self.generate_example()

    def generate_quiz(self) -> ActionResult:
        return self.generate_example()

    def evaluate_concept(self, action, concept=None, idx=None):
        if concept is None:
            return int(action[0] in self.numbers_inside)

        return concept.check(action[0])

    def gen_readable_format(self, result, show_answer=True):
        if show_answer:
            return "Number {:d}: {:s}".format(result[0], "yes" if result[1] == 1 else "no")
        else:
            return "Number {:d}: ?".format(result[0])

    def format_response(self, response):
        resp_map = {
            'yes': 1,
            'no': 0,
            'y': 1,
            'n': 0,
            '0': 0,
            '1': 1,
            0: 0,
            1: 1,
            True: 1,
            False: 0
        }
        if resp_map.get(response) is not None:
            response = resp_map[response]
        else:
            print("Unknown response {}, please answer with '[y]es/[n]o or 1/0".format(response))
            response = None

        return response

    def get_rl_actions(self, sample_count=None):
        return self.range

    def get_observation_space(self):
        return [0, 1]


class NumberGameConcept(ConceptItemBase):
    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def __init__(self, odd=False, even=False, square=False, cube=False, primes=False, multiples=False, powers=False,
                 ending=False, interval_start=False, interval_end=False, multiples_mod=False):
        self.multiples_mod = multiples_mod
        self.interval_end = interval_end
        self.interval_start = interval_start
        self.powers = powers
        self.multiples = multiples
        self.primes = primes
        self.cube = cube
        self.square = square
        self.even = even
        self.odd = odd
        self.ending = ending

    def check(self, number: int) -> any:
        if self.odd:
            result = self.odd_check(number)
        elif self.even:
            result = self.even_check(number)
        elif self.square:
            result = self.square_check(number)
        elif self.cube:
            result = self.cube_check(number)
        elif self.primes:
            result = self.primes_check(number)
        elif self.multiples_mod:
            result = self.multiples_mod_check(number)
        elif self.multiples:
            result = self.multiples_check(number)
        elif self.powers:
            result = self.powers_check(number)
        elif self.interval_start and self.interval_end:
            result = self.interval_check(number)
        elif self.ending:
            result = self.ending_check(number)
        else:
            result = False

        return int(result)

    def odd_check(self, number: int):
        return number % 2 == 1

    def even_check(self, number: int):
        return number % 2 == 0

    def square_check(self, number: int):
        return math.sqrt(number).is_integer()

    def cube_check(self, number: int):
        return (number**(1/3)).is_integer()

    def primes_check(self, number: int):
        return number in self.prime_numbers

    def multiples_check(self, number: int):
        return number % self.multiples == 0

    def multiples_mod_check(self, number: int):
        return number % self.multiples == (self.multiples + self.multiples_mod)

    def powers_check(self, number: int):
        return (math.log(number) / math.log(self.powers)).is_integer()

    def interval_check(self, number: int):
        return self.interval_start <= number <= self.interval_end

    def ending_check(self, number: int):
        return number % 10 == self.ending

    def __eq__(self, other):
        return (self.multiples_mod == other.multiples_mod
                and self.interval_end == other.interval_end
                and self.interval_start == other.interval_start
                and self.powers == other.powers
                and self.multiples == other.multiples
                and self.primes == other.primes
                and self.cube == other.cube
                and self.square == other.square
                and self.even == other.even
                and self.odd == other.odd
                and self.ending == other.ending)

    def __str__(self):
        if self.odd:
            return "odd numbers"
        if self.even:
            return "even numbers"
        if self.square:
            return "square numbers"
        if self.cube:
            return "cube numbers"
        if self.primes:
            return "prime numbers"
        if self.multiples_mod:
            return "multiples of {:d}{:d}".format(self.multiples, self.multiples_mod)
        if self.multiples:
            return "multiples of {:d}".format(self.multiples)
        if self.powers:
            return "powers of {:d}".format(self.powers)
        if self.interval_start and self.interval_end:
            return "numbers between {:d}-{:d}".format(self.interval_start, self.interval_end)
        if self.ending:
            return "numbers ending in {:d}".format(self.ending)
