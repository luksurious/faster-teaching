from abc import ABC, abstractmethod
from typing import Tuple

ActionResult = Tuple[any, str]


class ConceptBase(ABC):

    @abstractmethod
    def assess(self) -> bool:
        return False

    @abstractmethod
    def get_true_concepts(self):
        return None

    @abstractmethod
    def get_concept_space(self) -> iter:
        pass

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
    def evaluate_concept(self, action, concept):
        pass
