from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from actions import Actions

ActionResult = Tuple[any, any]


class ConceptItemBase(ABC):
    @abstractmethod
    def check(self, item) -> any:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __str__(self):
        pass


ConceptList = List[ConceptItemBase]


class ConceptBase(ABC):
    def __init__(self, action_costs: dict) -> None:
        self.action_costs = action_costs

    @abstractmethod
    def assess(self, learner) -> (bool, float):
        return False

    @abstractmethod
    def get_true_concept_idx(self) -> int:
        return -1

    @abstractmethod
    def get_concept_space(self) -> ConceptList:
        pass

    @abstractmethod
    def get_default_prior(self) -> np.ndarray:
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
    def generate_example(self) -> ActionResult:
        raw_result = []
        result = ""
        return raw_result, result

    @abstractmethod
    def generate_question_with_feedback(self) -> ActionResult:
        raw_result = []
        result = ""
        return raw_result, result

    @abstractmethod
    def generate_quiz(self) -> ActionResult:
        raw_result = []
        result = ""
        return raw_result, result

    @abstractmethod
    def evaluate_concept(self, action: any, concept=None, idx: int = None):
        pass

    @abstractmethod
    def gen_readable_format(self, result: ActionResult, show_answer=True):
        pass

    @abstractmethod
    def format_response(self, response: str) -> any:
        return response

    @abstractmethod
    def get_rl_actions(self) -> iter:
        pass

    @abstractmethod
    def get_observation_space(self) -> iter:
        pass
