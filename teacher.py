import numpy as np
import random
from concepts.concept_base import ConceptBase

random.seed()
# np.random.seed()


class Teacher:
    ACTION_EXAMPLE = 'Example'
    ACTION_QUIZ = 'Quiz'
    ACTION_QUESTION = 'Question with Feedback'

    def __init__(self, concept: ConceptBase, learning_phase_len: int = 3, max_actions: int = 40):
        self.learning_phase_len = learning_phase_len
        self.max_actions = max_actions

        # for memoryless model
        self.transition_noise = 0.15
        self.production_noise = 0.019

        self.concept = concept

        self.actions = {self.ACTION_EXAMPLE: self.concept.generate_example,
                        self.ACTION_QUIZ: self.concept.generate_quiz,
                        self.ACTION_QUESTION: self.concept.generate_question_with_feedback}

        self.concept_space = np.array(self.concept.get_concept_space())

        # uniform prior distribution
        self.concept_space_size = len(self.concept_space)
        self.prior_distribution = np.ones(self.concept_space_size) / self.concept_space_size
        self.belief_state = self.prior_distribution.copy()

        # position of true concept
        self.true_concept_pos = np.argmax(self.concept_space == self.concept.get_true_concepts())

    def teach(self):
        shown_concepts = []

        for action_num in range(self.max_actions):
            type, result, output = self.choose_action(shown_concepts)

            if type == self.ACTION_EXAMPLE:
                print("Let's see an example")
                response = None
                print(output)
            elif type == self.ACTION_QUIZ:
                print("Can you answer this quiz?")
                response = input(output)

                correct = response == result[1]
            else:
                # Question with feedback
                print("Question:")
                response = input(output)

                correct = response == result[1]
                if correct:
                    print("Yes, that's correct")
                else:
                    print("Not quite, the correct answer is %d" % result[1])
            self.update_belief(type, result, response)

            print("Current likely concepts: %d" % np.count_nonzero(self.belief_state > np.min(self.belief_state)))

            print("Contains correct concept?", self.belief_state[self.true_concept_pos] > np.min(self.belief_state))

            input("Continue?")

            if (action_num + 1) % self.learning_phase_len == 0:
                shown_concepts = []
                if self.assess():
                    return True

        return False

    def choose_action(self, shown_concepts):
        # random strategy
        current_type = random.sample(self.actions.keys(), 1)[0]

        result, output = self.actions[current_type]()
        while result in shown_concepts:
            result, output = self.actions[current_type]()

        shown_concepts.append(result)

        return current_type, result, output

    def assess(self):
        # assessment time
        print("Do you know the answers?")
        correct = self.concept.assess()

        if correct:
            print("Great! You learned all characters correctly")
            return True
        else:
            print("Nice try but there are some errors. Let's review some more...")
            return False

    def reveal_answer(self):
        print(self.concept.get_true_concepts())

    def update_belief(self, type, result, response):
        if type != self.ACTION_QUIZ and np.random.random() <= self.transition_noise:
            # transition noise probability: no state change;
            return

        new_belief = self.calc_unscaled_belief(type, result, response)

        # TODO does it make sense?
        if sum(new_belief) == 0:
            # quiz response inconsistent with previous state, calc only based on quiz now
            print("Inconsistent quiz response")
            self.belief_state[:] = 1
            new_belief = self.calc_unscaled_belief(type, result, response)

            if sum(new_belief) == 0:
                # still 0 means, invalid response; reset to prior probs
                new_belief = self.prior_distribution

        new_belief /= sum(new_belief)

        # is prior updated in every step??
        self.belief_state = new_belief

    def calc_unscaled_belief(self, type, result, response):
        new_belief = np.zeros_like(self.belief_state)
        for i, concept in enumerate(self.concept_space):
            concept_val = self.concept.evaluate_concept(result, concept)

            p_s = 0
            p_z = 0
            if type == self.ACTION_EXAMPLE:
                # TODO do I need to calculate the probability of the learners concept
                #  being already consistent with the new action shomehow?

                # TODO does prior from the paper here refer to the initial prior,
                #  or the prior previous to the current action?
                p_s = self.prior_distribution[i]
                if concept_val == result[1]:
                    p_z = 1-self.production_noise
                else:
                    p_z = self.production_noise
            elif type == self.ACTION_QUIZ:
                if concept_val == int(response) and self.belief_state[i] > 0:
                    # the true state of the learner doesn't change. but we can better infer which state he is in now
                    p_s = self.prior_distribution[i]
                    p_z = 1
            else:
                # TODO: not sure about this, but otherwise it doesnt make sense
                #  the observation is from the previous state, not from the next state
                #  type question with answer; or should somehow be taken into account that more likely now are
                #  concepts whith overlap in old and new state?
                p_s = self.prior_distribution[i]
                if concept_val == result[1]:
                    p_z = 1-self.production_noise
                else:
                    p_z = self.production_noise

            new_belief[i] = p_z * p_s

        return new_belief

