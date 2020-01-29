from concepts.concept_base import ConceptBase

from actions import Actions
from learner_models.base_belief import BaseBelief
from planners.base_planner import BasePlanner


class Teacher:
    def __init__(self, concept: ConceptBase, belief: BaseBelief, planner: BasePlanner,
                 learning_phase_len: int = 3, max_phases: int = 40, verbose: bool = False):

        self.learning_phase_len = learning_phase_len
        self.max_phases = max_phases

        self.planner = planner

        self.learner = None

        self.action_history = []
        self.assessment_history = []
        self.verbose = verbose

        self.concept = concept
        self.belief = belief

    def reset(self):
        self.action_history = []
        self.assessment_history = []
        self.planner.reset()
        self.belief.reset()

    def teach(self):
        for self.action_count in range(self.max_phases*3):
            action_type, equation, result = self.planner.choose_action()

            action_data = (equation, result)
            action_data_hidden = (equation, None)

            if action_type == Actions.EXAMPLE:
                if self.verbose:
                    print("Let's see an example")
                self.learner.see_example(action_data)
                response = None
            elif action_type == Actions.QUIZ:
                if self.verbose:
                    print("Can you answer this quiz?")
                response = self.learner.see_quiz(action_data_hidden)
                action_data = action_data_hidden
            else:
                # Question with feedback
                if self.verbose:
                    print("Question:")
                response = self.learner.see_question_question(action_data_hidden)

                correct = response == action_data[1]

                self.learner.see_question_feedback(action_data, correct)

            self.belief.update_belief(action_type, action_data, response)

            self.learner.finish_action(action_data)

            self.action_history.append((action_type, action_data))

            if (self.action_count + 1) % self.learning_phase_len == 0:
                if self.assess():
                    return True

        return False

    def assess(self):
        # assessment time
        if self.verbose:
            print("Do you know the answers?")
        correct, errors = self.concept.assess(self.learner)

        self.assessment_history.append(errors)

        if correct:
            if self.verbose:
                print("Great! You learned all characters correctly")
            return True
        else:
            if self.verbose:
                print("Nice try but there are some errors. Let's review some more...")
            return False

    def reveal_answer(self):
        print("True answer:")
        print(self.concept.get_true_concepts())

    def enroll_learner(self, learner):
        self.learner = learner
