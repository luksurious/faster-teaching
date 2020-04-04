from concepts.concept_base import ConceptBase
from learners.base_learner import BaseLearner
import time


class HumanLearner(BaseLearner):
    def __init__(self, concept: ConceptBase):
        super().__init__(concept)

        self.total_time = 0
        self.action_start_time = 0

    def see_example(self, example):
        self.action_start_time = time.time()
        print(self.concept.gen_readable_format(example))

    def see_quiz(self, quiz):
        self.action_start_time = time.time()
        response = input(self.concept.gen_readable_format(quiz, False))
        response = self.concept.format_response(response)
        return response

    def see_question_question(self, question):
        self.action_start_time = time.time()
        # Question with feedback
        response = input(self.concept.gen_readable_format(question, False))
        response = self.concept.format_response(response)

        return response

    def see_question_feedback(self, question, correct):
        if correct:
            print("Yes, that's correct")
        else:
            print("Not quite, the correct answer is %d" % question[1])

    def answer(self, item):
        curr_guess = input("What is %s?" % item[1])
        curr_guess = self.concept.format_response(curr_guess)

        return curr_guess

    def finish_action(self, action):
        input("Continue?")

        self.total_time += time.time() - self.action_start_time
        self.action_start_time = 0

