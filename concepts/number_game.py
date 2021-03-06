import math

import numpy as np

from actions import ACTION_COSTS_SAMPLE, Actions
from concepts.concept_base import ConceptBase, ActionResult, ConceptItemBase
from random_ng import rand_ng


class NumberGame(ConceptBase):
    ACTION_COSTS = {
        Actions.EXAMPLE: 2.4,
        Actions.QUIZ: 2.8,
        Actions.FEEDBACK: 4.8
    }
    TRANS_NOISE = {
        'memoryless': 0.14,
        'discrete': 0.10,
        'continuous': 0.15,
    }
    PROD_NOISE = {
        'memoryless': 0.25,
        'discrete': 0.18,
        'continuous': 0.21,
    }

    def __init__(self, target_concept='mul7', space_mode='default'):
        self.range = range(1, 101)

        # self.prior_lambda = 2/3  # lambda from some other paper
        if space_mode == 'orig':
            self.prior_lambda = 0.55519  # lambda best matching orig priors but too strange
        else:
            self.prior_lambda = 1/2  # lambda from paper 2000
        self.erlang_sigma = 10

        self.inside_prob = 0.5

        self.concept_space, self.prior = self.generate_plausible_concepts(space_mode)

        if target_concept == 'mul4-1':
            self.cur_concept = NumberGameConcept(multiples=4, multiples_mod=-1)
        elif target_concept == '64-83':
            self.cur_concept = NumberGameConcept(interval_start=64, interval_end=83)
        elif target_concept == 'mul7':
            self.cur_concept = NumberGameConcept(multiples=7)
        else:
            raise ValueError('unknown target concept %s' % target_concept)

        self.true_concept_idx = self.find_true_concept_idx()

        super().__init__()

    def find_true_concept_idx(self):
        for idx, concept in enumerate(self.concept_space):
            if concept == self.cur_concept:
                return idx

        return -1

    def generate_plausible_concepts(self, space_mode='default'):
        math_concepts, range_concepts, mod_math_concepts = [], [], []
        range_priors = []

        math_concepts.append(NumberGameConcept(odd=True))
        math_concepts.append(NumberGameConcept(even=True))
        math_concepts.append(NumberGameConcept(square=True))
        math_concepts.append(NumberGameConcept(cube=True))
        math_concepts.append(NumberGameConcept(primes=True))

        for i in range(3, 51):
            # multiples of 3-12 are three times present in the orig concept space
            # bigger multiples "only" twice
            if i <= 12:
                math_concepts.append(NumberGameConcept(multiples=i))
                if space_mode == 'orig':
                    mod_math_concepts.append(NumberGameConcept(multiples=i))
                    mod_math_concepts.append(NumberGameConcept(multiples=i))
            else:
                mod_math_concepts.append(NumberGameConcept(multiples=i))
                if space_mode == 'orig':
                    mod_math_concepts.append(NumberGameConcept(multiples=i))

            for mod in range(1, i):
                mod_math_concepts.append(NumberGameConcept(multiples=i, multiples_mod=-mod))

        for i in range(2, 11):
            math_concepts.append(NumberGameConcept(powers=i))
            math_concepts.append(NumberGameConcept(powers=i, powers_zero=True))

        for i in range(1, 10):
            math_concepts.append(NumberGameConcept(ending=i))

        for a in range(1, 101):
            for b in range(a, 101):
                range_concepts.append(NumberGameConcept(interval_start=a, interval_end=b))
                range_size = (b - a + 1)
                range_priors.append(range_size / self.erlang_sigma**2 * math.exp(-range_size/self.erlang_sigma))

        math_count = len(math_concepts)
        range_count = len(range_concepts)
        mod_math_count = len(mod_math_concepts)

        math_priors = [self.prior_lambda / 2 / math_count] * math_count

        # range_prior_share = (1-self.prior_lambda) * range_count / (range_count+mod_math_count)
        range_prior_share = (1-self.prior_lambda)
        # mod_math_prior_share = (1-self.prior_lambda) * mod_math_count / (range_count+mod_math_count)

        range_priors = np.array(range_priors)
        range_priors /= np.sum(range_priors)

        range_priors = np.array(range_priors) * range_prior_share
        mod_math_priors = [self.prior_lambda / 2 / mod_math_count] * mod_math_count
        # [mod_math_prior_share / mod_math_count] * mod_math_count

        concepts = math_concepts + mod_math_concepts + range_concepts
        priors = np.concatenate([math_priors, mod_math_priors, range_priors])
        # priors /= np.sum(priors)

        # print("Math concepts: {:d}, mod math {:d}, range concepts: {:d}".format(
        #     math_count, mod_math_count, range_count
        # ))

        return concepts, priors

    def get_default_prior(self) -> iter:
        return self.prior

    def assess(self, learner) -> (bool, float):
        guesses = []
        correct = True
        errors = 0

        cor_numbers = self.sample_inside(5)
        incor_numbers = self.sample_outside(5)

        questions = np.concatenate([cor_numbers, incor_numbers])

        rand_ng.rg.shuffle(questions)

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
        cor_numbers = rand_ng.rg.choice(self.cur_concept.numbers_inside, number, replace=False)
        cor_numbers = np.stack([cor_numbers, [1] * number], axis=1)

        return cor_numbers

    def sample_outside(self, number: int):
        incor_numbers = rand_ng.rg.choice(self.cur_concept.numbers_outside, number, replace=False)
        incor_numbers = np.stack([incor_numbers, [0] * number], axis=1)

        return incor_numbers

    def get_true_concept_idx(self):
        return self.true_concept_idx

    def get_concept_space(self) -> iter:
        return self.concept_space

    def generate_example(self) -> ActionResult:
        if rand_ng.rg.random() > self.inside_prob:
            return rand_ng.rg.choice(self.cur_concept.numbers_inside), True
        else:
            return rand_ng.rg.choice(self.cur_concept.numbers_outside), False

    def generate_question_with_feedback(self) -> ActionResult:
        return self.generate_example()

    def generate_quiz(self) -> ActionResult:
        return self.generate_example()

    def evaluate_concept(self, action, concept=None, idx=None):
        if concept is None:
            return int(action in self.cur_concept.numbers_inside)

        return int(action in concept.numbers_inside)

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
                 powers_zero=False,
                 ending=False, interval_start=False, interval_end=False,
                 multiples_mod=False, multiples_start=0):
        self.multiples_start = multiples_start
        self.multiples_mod = multiples_mod
        self.interval_end = interval_end
        self.interval_start = interval_start
        self.powers = powers
        self.powers_zero = powers_zero
        self.multiples = multiples
        self.primes = primes
        self.cube = cube
        self.square = square
        self.even = even
        self.odd = odd
        self.ending = ending

        self.range = range(1, 101)
        self.numbers_inside, self.numbers_outside = [], []

        for i in self.range:
            if self.check(i):
                self.numbers_inside.append(i)
            else:
                self.numbers_outside.append(i)

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
        cbrt = np.cbrt(number).round(5)
        return cbrt.is_integer() and int(cbrt)**3 == number

    def primes_check(self, number: int):
        return number in self.prime_numbers

    def multiples_check(self, number: int):
        if self.multiples_start > 0:
            return number % self.multiples == self.multiples_start
        return number % self.multiples == 0

    def multiples_mod_check(self, number: int):
        if self.multiples_mod < 0:
            return number % self.multiples == (self.multiples + self.multiples_mod)
        else:
            return number % self.multiples == self.multiples_mod

    def powers_check(self, number: int):
        power_of = (math.log(number) / math.log(self.powers))
        return power_of.is_integer() and (power_of > 0 or self.powers_zero)

    def interval_check(self, number: int):
        return self.interval_start <= number <= self.interval_end

    def ending_check(self, number: int):
        return number % 10 == self.ending

    def __eq__(self, other):
        return (self.multiples_mod == other.multiples_mod
                and self.interval_end == other.interval_end
                and self.interval_start == other.interval_start
                and self.powers == other.powers
                and self.powers_zero == other.powers_zero
                and self.multiples == other.multiples
                and self.multiples_start == other.multiples_start
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
            return "multiples of {:d} {:+d}".format(self.multiples, self.multiples_mod)
        if self.multiples and self.multiples_start > 0:
            return "multiples of {:d} starting at {:d}".format(self.multiples, self.multiples_start)
        if self.multiples:
            return "multiples of {:d}".format(self.multiples)
        if self.powers and self.powers_zero:
            return "powers of {:d} (incl. ^0)".format(self.powers)
        if self.powers and not self.powers_zero:
            return "powers of {:d} (excl. ^0)".format(self.powers)
        if self.interval_start and self.interval_end:
            return "numbers between {:d}-{:d}".format(self.interval_start, self.interval_end)
        if self.ending:
            return "numbers ending in {:d}".format(self.ending)


if __name__ == '__main__':
    # ng = NumberGame()
    #
    # print(len(ng.concept_space))

    import time
    start_time = time.time()

    with open('../numberHyp.txt') as f:
        orig_hyps = f.read()
    orig_hyps = np.array([line.split('\t') for line in orig_hyps.splitlines()], dtype=int)

    with open('../prior.txt') as f:
        orig_priors = f.read()
    orig_priors = list(map(float, orig_priors.splitlines()))

    print("orig priors sum", np.sum(orig_priors))

    number_game = NumberGame()

    print("Total concepts", len(number_game.concept_space))

    for own_idx, own_hyp in enumerate(number_game.concept_space):
        # if own_hyp.interval_start is False:
        #     continue
        # if own_hyp.multiples not in [4, 7]:# and (own_hyp.interval_start != 64 and own_hyp.interval_end != 83):
        #     continue
        # if own_hyp.interval_start is False or (own_hyp.interval_start != 1 and own_hyp.interval_end != 100):
        #     continue
        hyp_nums = np.zeros(100, dtype=int)

        if len(own_hyp.numbers_inside) == 0:
            print('###### empty concept', str(own_hyp))
            continue

        hyp_nums[np.array(own_hyp.numbers_inside)-1] = 1

        found = False
        for orig_idx, orig_hyp in enumerate(orig_hyps):
            if np.all(orig_hyp == hyp_nums):
                own_prior = number_game.prior[own_idx]
                orig_prior = orig_priors[orig_idx]

                orig_hyps = np.delete(orig_hyps, orig_idx, 0)
                orig_priors = np.delete(orig_priors, orig_idx, 0)
                found = True
                if len(orig_hyps) % 500 == 0:
                    print('found', str(own_hyp), "left", len(orig_hyps))

                if not np.isclose(own_prior, orig_prior, atol=1e-8):  # and (orig_prior > .0066 or own_prior > 0.0001):
                    print('Priors for {}: orig {:.7f} vs {:.7f}, delta {:.7f}'.format(
                        str(own_hyp), orig_prior, own_prior, abs(orig_prior-own_prior)
                    ))

                break

        if not found:
            print('##### hyp not in orig', str(own_hyp))

    print("Time elapsed", time.time()-start_time)

    print("unmatched hyps", len(orig_hyps))
    print("unmatched priors", orig_priors)
    # print(orig_hyps)

    # with open('number_hyps_new.csv', 'w') as f:
    #     f.writelines([
    #         ",".join(map(str, hyp)).replace('0,', ',') + "\n" for hyp in orig_hyps.tolist()
    #     ])
