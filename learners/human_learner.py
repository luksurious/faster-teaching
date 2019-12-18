from concepts.concept_base import ConceptBase
from learners.base_learner import BaseLearner


class HumanLearner(BaseLearner):
    def __init__(self, concept: ConceptBase):
        super().__init__(concept)

        self.problem_len = len(concept.get_true_concepts())
        # self.letters = [chr(ord('A')+i) for i in range(self.problem_len)]

    def see_example(self, example):
        print(self.concept.gen_readable_format(example))

    def see_quiz(self, quiz):
        response = input(self.concept.gen_readable_format(quiz, False))
        return response

    def see_question(self, question):
        # Question with feedback
        response = input(self.concept.gen_readable_format(question, False))
        try:
            response = int(response)
        except:
            response = -1

        return response

    def see_question_feedback(self, question, correct):
        if correct:
            print("Yes, that's correct")
        else:
            print("Not quite, the correct answer is %d" % question[1])

    def answer(self, item):
        curr_guess = input("What is %s?" % item[1])

        return curr_guess

    def finish_action(self):
        input("Continue?")

