from abc import ABC, abstractmethod

from concepts.concept_base import ConceptBase


class BasePlanner(ABC):
    def __init__(self, concept: ConceptBase, actions: list):
        self.actions = actions
        self.concept = concept

    @abstractmethod
    def choose_action(self):
        pass

    @abstractmethod
    def start_teaching_phase(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def perform_preplanning(self):
        pass

    @abstractmethod
    def load_preplanning(self, data):
        pass