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

        self.concept = concept

        self.actions = {self.ACTION_EXAMPLE: self.concept.generate_example,
                        self.ACTION_QUIZ: self.concept.generate_quiz,
                        self.ACTION_QUESTION: self.concept.generate_question_with_feedback}

    def teach(self):
        shown_concepts = []

        for action_num in range(self.max_actions):
            type, result, output = self.choose_action(shown_concepts)

            if type == self.ACTION_EXAMPLE:
                print("Let's see an example")
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
