from copy import deepcopy

from learner_models.base_belief import BaseBelief
from concepts.concept_base import ConceptBase
from actions import Actions

import math
import numpy as np


# TODO: verify recreation of particles based on history
#  visualize evolution of beliefs
#  adapt belief base
class ContinuousModel(BaseBelief):

    def __init__(self, belief_state, prior, concept: ConceptBase, particle_num: int = 16, verbose: bool = True):
        super().__init__(belief_state, prior, concept, verbose)

        self.transition_noise = 0.14  # / self.prior_pos_len
        self.production_noise = 0.12

        self.particle_num = particle_num

        self.particle_dists = []
        self.particle_weights = []
        self.init_particles(prior)

        self.action_history = []

        self.particle_depletion_limit = 0.005

        self.production_noise_per_response = self.production_noise / len(concept.get_observation_space())

        # pre-calculate state-action concept values
        # TODO duplicated as in letter addition?
        self.state_action_values = {}
        self.pre_calc_state_values()

    def init_particles(self, prior):
        # init particles
        particle1_dist = np.copy(prior)
        particle1_weight = 1
        self.particle_dists.append(particle1_dist)
        self.particle_weights.append(particle1_weight)

        # TODO if prior is not uniform
        # TODO verify
        # particle2_dist = prior.copy()
        # particle2_weight = 0.5

    def pre_calc_state_values(self):
        for action in self.concept.get_rl_actions():
            self.state_action_values[action] = np.zeros(len(self.states))
            for idx, state in enumerate(self.states):
                self.state_action_values[action][idx] = self.concept.evaluate_concept((action,), state, idx)

    def update_belief(self, action_type, result, response):
        self.action_history.append((action_type, result, response))

        if response is not None:
            # update based on response
            self.update_from_response(response, result)

        if action_type != Actions.QUIZ:
            # update based on content
            self.update_from_content(result)

    def update_from_content(self, result):
        concepts_inconsistent = self.state_action_values[result[0]] != result[1]

        new_particles = []
        new_particle_weights = []
        for idx, particle in enumerate(self.particle_dists):
            particle_weight = self.particle_weights[idx]

            # particle for not being transitioned
            non_transition_particle = np.copy(particle)
            non_transition_weight = particle_weight * self.transition_noise
            new_particles.append(non_transition_particle)
            new_particle_weights.append(non_transition_weight)

            particle[concepts_inconsistent] = 0
            particle /= np.sum(particle)
            transitioned_weight = particle_weight * (1 - self.transition_noise)
            new_particles.append(particle)
            new_particle_weights.append(transitioned_weight)

        self.particle_dists = new_particles
        self.particle_weights = new_particle_weights

        # check for particle depletion
        # TODO verify sum instead of max
        if np.sum(self.particle_weights) < self.particle_depletion_limit:
            self.recreate_particles()

        if len(self.particle_dists) > self.particle_num:

            while len(self.particle_dists) > self.particle_num:
                min_idx = np.argmin(self.particle_weights)
                del self.particle_dists[min_idx]
                del self.particle_weights[min_idx]

        # re-normalize weights
        weight_sum = np.sum(self.particle_weights)
        self.particle_weights = [w / weight_sum for w in self.particle_weights]

    def update_from_response(self, response, result):
        concepts_w_val = self.state_action_values[result[0]] == response

        for idx, particle in enumerate(self.particle_dists):
            current_weight = self.particle_weights[idx]

            p_z = np.sum(particle[concepts_w_val])
            new_weight = current_weight * ((1 - self.production_noise) * p_z + self.production_noise_per_response)

            self.particle_weights[idx] = new_weight

        # check for particle depletion
        if max(self.particle_weights) < self.particle_depletion_limit:
            self.recreate_particles()
        else:
            # normalize
            weight_sum = np.sum(self.particle_weights)
            self.particle_weights = [w / weight_sum for w in self.particle_weights]

    def recreate_particles(self):
        self.particle_dists = []
        self.particle_weights = []
        particle1_dist = np.copy(self.prior)
        particle1_weight = 0.5
        self.particle_dists.append(particle1_dist)
        self.particle_weights.append(particle1_weight)

        # particle 2: consistent with observed data
        particle2_dist = np.copy(self.prior)
        particle2_weight = 0.5

        for action_type, result, response in self.action_history:
            # update according to observed data
            # TODO: Q: does that mean only taking examples into account (i.e. observed data?)
            particle2_dist = self.transition_model(particle2_dist, None, action_type, result, None)

        self.particle_dists.append(particle2_dist)
        self.particle_weights.append(particle2_weight)

    def observation_model(self, observation, new_state, action_type, action, concept_val):
        concepts_w_val = self.state_action_values[action[0]] == observation

        p_z = np.sum(new_state[concepts_w_val])

        return p_z

    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        concepts_inconsistent = self.state_action_values[action[0]] != action[1]
        new_state[concepts_inconsistent] = 0

        new_state /= np.sum(new_state)

        return new_state

    def get_concept_prob(self, index):
        prob = 0

        for idx, particle in enumerate(self.particle_dists):
            prob += self.particle_weights[idx] * particle[index]

        return prob

    def get_state(self):
        return deepcopy(self.particle_dists), self.particle_weights.copy(), self.action_history.copy()

    def set_state(self, state):
        self.particle_dists = deepcopy(state[0])
        self.particle_weights = state[1].copy()
        self.action_history = state[2].copy()

    def reset(self):
        # super().reset()
        self.particle_dists = []
        self.particle_weights = []
        self.init_particles(self.prior)

        self.action_history = []

    def __copy__(self):
        state = self.get_state()
        new_model = ContinuousModel(self.belief_state.copy(), self.prior, self.concept, self.particle_num, self.verbose)
        new_model.set_state(state)

        return new_model
