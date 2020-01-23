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

        self.particle_dists = np.zeros((2*particle_num, len(self.states)))
        self.particle_weights = np.zeros(2*particle_num)
        self.valid_particles = np.array([False]*particle_num*2)

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
        self.particle_dists[0] = particle1_dist
        self.particle_weights[0] = particle1_weight
        self.valid_particles[0] = True

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
        fillable_slots = np.argwhere(self.valid_particles == False).ravel().tolist()

        valid_indices = np.argwhere(self.valid_particles).ravel()
        for idx in valid_indices:
            particle = self.particle_dists[idx]
            particle_weight = self.particle_weights[idx]

            # new particle if transitioned
            transitioned_particle = np.copy(particle)
            transitioned_particle[concepts_inconsistent] = 0
            transitioned_particle /= np.sum(transitioned_particle)
            transitioned_weight = particle_weight * (1 - self.transition_noise)

            new_idx = fillable_slots.pop(0)
            self.particle_dists[new_idx] = transitioned_particle
            self.particle_weights[new_idx] = transitioned_weight
            self.valid_particles[new_idx] = True

            # particle for not being transitioned stays the same
            self.particle_weights[idx] = particle_weight * self.transition_noise

        # check for particle depletion
        # TODO verify: sum here instead of max
        weight_sum = np.sum(self.particle_weights[self.valid_particles])
        if weight_sum < self.particle_depletion_limit:
            self.recreate_particles()

        if np.count_nonzero(self.valid_particles) > self.particle_num:
            remove_count = np.count_nonzero(self.valid_particles) - self.particle_num

            lowest_idx = np.argpartition(self.particle_weights, -remove_count)
            self.valid_particles[lowest_idx] = False
            self.particle_weights[lowest_idx] = 0
        # re-normalize weights
        self.particle_weights[self.valid_particles] /= np.sum(self.particle_weights[self.valid_particles])

    def update_from_response(self, response, result):
        concepts_w_val = self.state_action_values[result[0]] == response
        valid_indices = np.argwhere(self.valid_particles).ravel()
        for idx in valid_indices:
            current_weight = self.particle_weights[idx]

            p_z = np.sum(self.particle_dists[idx][concepts_w_val])
            new_weight = current_weight * ((1 - self.production_noise) * p_z + self.production_noise_per_response)

            self.particle_weights[idx] = new_weight
        # check for particle depletion
        if np.max(self.particle_weights) < self.particle_depletion_limit:
            self.recreate_particles()
        else:
            # normalize
            # TODO paper remark: checking for depletion before re-normalizing?
            self.particle_weights /= np.sum(self.particle_weights)

    def recreate_particles(self):
        self.particle_dists = np.zeros_like(self.particle_dists)
        self.particle_weights = np.zeros_like(self.particle_weights)
        self.valid_particles = np.array([False]*self.particle_num*2)

        particle1_dist = np.copy(self.prior)
        particle1_weight = 0.5
        self.particle_dists[0] = particle1_dist
        self.particle_weights[0] = particle1_weight
        self.valid_particles[0] = True

        # particle 2: consistent with observed data
        particle2_dist = np.copy(self.prior)
        particle2_weight = 0.5

        for action_type, result, response in self.action_history:
            # update according to observed data
            # TODO: Q: does that mean only taking examples into account (i.e. observed data?)
            particle2_dist = self.transition_model(particle2_dist, None, action_type, result, None)

        self.particle_dists[1] = particle2_dist
        self.particle_weights[1] = particle2_weight
        self.valid_particles[1] = True

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

        for idx in np.argwhere(self.valid_particles).ravel():
            prob += self.particle_weights[idx] * self.particle_dists[idx][index]

        return prob

    def get_state(self):
        return np.copy(self.particle_dists), np.copy(self.particle_weights), np.copy(self.valid_particles),\
               self.action_history.copy()

    def set_state(self, state):
        self.particle_dists = np.copy(state[0])
        self.particle_weights = np.copy(state[1])
        self.valid_particles = np.copy(state[2])
        self.action_history = state[3].copy()

    def reset(self):
        self.particle_dists = np.zeros((2*self.particle_num, len(self.states)))
        self.particle_weights = np.zeros(2*self.particle_num)
        self.valid_particles = np.array([False]*self.particle_num*2)

        self.init_particles(self.prior)

        self.action_history = []

    def __copy__(self):
        state = self.get_state()
        new_model = ContinuousModel(self.belief_state.copy(), self.prior, self.concept, self.particle_num, self.verbose)
        new_model.set_state(state)

        return new_model
