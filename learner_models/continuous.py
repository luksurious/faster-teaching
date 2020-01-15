from learner_models.base_belief import BaseBelief
from concepts.concept_base import ConceptBase
from actions import Actions

import math
import numpy as np


class ContinuousModel(BaseBelief):

    def __init__(self, belief_state, prior, concept: ConceptBase, particle_num: int = 16, verbose: bool = True):
        super().__init__(belief_state, prior, concept, verbose)

        self.transition_noise = 0.14  # / self.prior_pos_len
        self.production_noise = 0.12

        self.particle_num = particle_num

        self.particle_dists = []
        self.particle_weights = []
        self.init_particles(prior)

        self.particle_depletion_limit = 0.005

        self.production_noise_per_response = self.production_noise / len(concept.get_observation_space())

        # pre-calculate state-action concept values
        # TODO duplicated as in letter addition?
        self.state_action_values = {}
        self.pre_calc_state_values()

        self.action_history = []

    def init_particles(self, prior):
        # init particles
        particle1_dist = prior.copy()
        particle1_weight = 1
        self.particle_dists.append(particle1_dist)
        self.particle_weights.append(particle1_weight)
        # TODO if prior is not uniform
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
            for idx, particle in enumerate(self.particle_dists):
                current_weight = self.particle_weights[idx]

                p_z = self.observation_model(response, particle, action_type, result, None)
                new_weight = current_weight * ((1 - self.production_noise) * p_z + self.production_noise_per_response)

                self.particle_weights[idx] = new_weight

            # check for particle depletion
            if max(self.particle_weights) < self.particle_depletion_limit:
                self.recreate_particles()
            else:
                # normalize
                weight_sum = np.sum(self.particle_weights)
                self.particle_weights = [w/weight_sum for w in self.particle_weights]

        if action_type != Actions.QUIZ:
            # update based on content

            new_particles = []
            new_particle_weights = []
            for idx, particle in enumerate(self.particle_dists):
                particle_weight = self.particle_weights[idx]

                # particle for not being transitioned
                non_transition_particle = particle.copy()
                non_transition_weight = particle_weight * self.transition_noise
                new_particles.append(non_transition_particle)
                new_particle_weights.append(non_transition_weight)

                transitioned_particle = self.transition_model(particle, idx, action_type, result, None)
                transitioned_weight = particle_weight * (1 - self.transition_noise)
                new_particles.append(transitioned_particle)
                new_particle_weights.append(transitioned_weight)

            self.particle_dists = new_particles
            self.particle_weights = new_particle_weights

            # check for particle depletion
            if max(self.particle_weights) < self.particle_depletion_limit:
                self.recreate_particles()

            if len(self.particle_dists) > self.particle_num:

                while len(self.particle_dists) > self.particle_num:
                    min_idx = np.argmin(self.particle_weights)
                    del self.particle_dists[min_idx]
                    del self.particle_weights[min_idx]

            # re-normalize weights
            weight_sum = np.sum(self.particle_weights)
            self.particle_weights = [w/weight_sum for w in self.particle_weights]

    def recreate_particles(self):
        self.particle_dists = []
        self.particle_weights = []
        particle1_dist = self.prior.copy()
        particle1_weight = 0.5
        self.particle_dists.append(particle1_dist)
        self.particle_weights.append(particle1_weight)

        # particle 2: consistent with observed data
        particle2_dist = self.prior.copy()
        particle2_weight = 0.5

        for action_type, result, response in self.action_history:
            # update according to observed data
            # TODO: Q: does that mean only taking examples into account (i.e. observed data?)
            pass

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

    def __copy__(self):
        return ContinuousModel(self.belief_state.copy(), self.prior, self.concept, self.particle_num, self.verbose)
