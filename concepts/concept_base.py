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
    def generate_example(self) -> ActionResult:
        raw_result = []
        output = ""
        return raw_result, output

    @abstractmethod
    def generate_question_with_feedback(self) -> ActionResult:
        raw_result = []
        output = ""
        return raw_result, output

    @abstractmethod
    def generate_quiz(self) -> ActionResult:
        raw_result = []
        output = ""
        return raw_result, output
