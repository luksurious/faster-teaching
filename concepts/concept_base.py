from abc import ABC, abstractmethod
from typing import Tuple

from actions import Actions

ActionResult = Tuple[any, str]


class ConceptBase(ABC):

    @abstractmethod
    def assess(self, learner) -> (bool, float):
        return False

    @abstractmethod
    def get_true_concepts(self):
        return None

    @abstractmethod
    def get_concept_space(self) -> iter:
        pass

    def teaching_action(self, action_type: Actions) -> ActionResult:
        if action_type == Actions.EXAMPLE:
            return self.generate_example()
        elif action_type == Actions.QUIZ:
            return self.generate_quiz()
        elif action_type == Actions.FEEDBACK:
            return self.generate_question_with_feedback()
        else:
            raise Exception("Unknown action %s" % str(action_type))

    @abstractmethod
    def generate_example(self, alternative_concept=None) -> ActionResult:
        raw_result = []
        output = ""
        return raw_result, output

    @abstractmethod
    def generate_question_with_feedback(self, alternative_concept=None) -> ActionResult:
        raw_result = []
        output = ""
        return raw_result, output

    @abstractmethod
    def generate_quiz(self, alternative_concept=None) -> ActionResult:
        raw_result = []
        output = ""
        return raw_result, output

    @abstractmethod
    def evaluate_concept(self, action, concept=None, idx=None):
        pass

    @abstractmethod
    def gen_readable_format(self, result, show_answer=True):
        pass

    @abstractmethod
    def get_rl_actions(self, sample_count = None):
        pass

    @abstractmethod
    def get_observation_space(self):
        pass
