import random

from concepts.concept_base import ConceptBase
from planners.base_planner import BasePlanner


class RandomPlanner(BasePlanner):
    def __init__(self, concept: ConceptBase, actions: list):
        super().__init__(concept, actions)

        self.shown_concepts = []

    def choose_action(self):
        # random strategy
        current_type = random.sample(self.actions, 1)[0]

        equation, result = self.concept.teaching_action(current_type)
        while equation in self.shown_concepts:
            equation, result = self.concept.teaching_action(current_type)

        # TODO check for different types
        self.shown_concepts.append(equation)

        return current_type, equation, result

    def start_teaching_phase(self):
        self.shown_concepts = []

    def reset(self):
        pass

    def perform_preplanning(self):
        pass

    def load_preplanning(self, data):
        pass