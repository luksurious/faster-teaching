from abc import ABC, abstractmethod

from concepts.concept_base import ConceptBase


class BaseLearner(ABC):
    def __init__(self, concept: ConceptBase):
        self.concept = concept

    @abstractmethod
    def see_example(self, example):
        pass

    @abstractmethod
    def see_quiz(self, quiz):
        pass

    @abstractmethod
    def see_question(self, question):
        pass

    @abstractmethod
    def see_question_feedback(self, question, correct):
        pass

    @abstractmethod
    def answer(self, item):
        pass

    def finish_action(self, action):
        pass
