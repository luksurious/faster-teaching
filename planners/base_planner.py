from abc import ABC, abstractmethod

from concepts.concept_base import ConceptBase


class BasePlanner(ABC):
    def __init__(self, concept: ConceptBase, actions: list):
        self.actions = actions
        self.concept = concept
        self.plan_duration_history = []
        self.pre_plan_duration = 0

    @abstractmethod
    def choose_action(self, prev_response=None):
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
