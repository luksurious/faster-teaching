from abc import ABC, abstractmethod

from concepts.concept_base import ConceptBase


class ModelBase(ABC):

    def __init__(self, states, prior, concept: ConceptBase):
        self.states = states
        self.prior = prior
        self.concept = concept
        self.transition_noise = 0
        self.production_noise = 0

    @abstractmethod
    def observation_model(self, observation, new_state, action_type, action):
        pass

    @abstractmethod
    def transition_model(self, new_state, new_idx, action_type, action):
        pass

    @abstractmethod
    def see_action(self, action_type, action):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass
