# import numpy as np
import random
from concepts.concept_base import ConceptBase

random.seed()


class Teacher:
    def __init__(self, concept: ConceptBase, learning_phase_len: int = 3, max_actions: int = 40):
        self.learning_phase_len = learning_phase_len
        self.max_actions = max_actions

        self.concept = concept

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
        # random strategy
        current_type = random.sample(
            [self.concept.generate_example, self.concept.generate_question_with_feedback, self.concept.generate_quiz],
            1)[0]

        result, output = current_type()
        while result in shown_concepts:
            result, output = current_type()

        shown_concepts.append(result)

        return result, output

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
