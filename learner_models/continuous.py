from copy import deepcopy

from learner_models.base_belief import BaseBelief
from concepts.concept_base import ConceptBase

import numpy as np


class ContinuousModel(BaseBelief):
    name = 'continuous'

    def __init__(self, prior, concept: ConceptBase, particle_num: int = 16, verbose: bool = True):
        super().__init__([], prior, concept, verbose=verbose)

        self.particle_num = particle_num

        self.particle_dists = []
        self.particle_weights = []
        self.init_particles()

        self.action_history = []

        self.particle_depletion_limit = 0.005

        self.history_calcs = 0

    def init_particles(self):
        # init particles
        # Uniform particle
        particle1_dist = np.ones(len(self.hypotheses)) / len(self.hypotheses)
        particle1_weight = 0.5
        self.particle_dists.append(particle1_dist)
        self.particle_weights.append(particle1_weight)

        # Prior particle
        particle2_dist = np.copy(self.prior)

        if np.allclose(particle2_dist, particle1_dist):
            self.particle_weights[0] = 1
            return

        particle2_weight = 0.5
        self.particle_dists.append(particle2_dist)
        self.particle_weights.append(particle2_weight)

    def update_belief(self, action_type, result, response):
        self.action_history.append((action_type, result, response))

        if response is not None:
            # update based on response
            self.update_from_response(response, result)

        # since the concepts are modeled as distributions, even for correct feedback actions we can use it to eliminate
        # probability on non-matching concepts
        transition_happened = result[1] is not None
        if transition_happened:
            # update based on content
            self.update_from_content(result)

    def update_from_content(self, result):
        concepts_inconsistent = self.state_action_values[result[0]] != result[1]

        new_particle_weights, new_particles = self.create_updated_particles(concepts_inconsistent)
        self.particle_dists = new_particles
        self.particle_weights = new_particle_weights

        self.check_particles_valid()

    def check_particles_valid(self):
        # check for particle depletion
        if np.sum(self.particle_weights) < self.particle_depletion_limit:
            self.recreate_particles()
        else:
            self.assert_particle_limit()

            # re-normalize weights
            weight_sum = np.sum(self.particle_weights)
            self.particle_weights = [w / weight_sum for w in self.particle_weights]

    def create_updated_particles(self, concepts_inconsistent):
        new_particles = []
        new_particle_weights = []

        for idx, particle in enumerate(self.particle_dists):
            particle_weight = self.particle_weights[idx]

            # particle for not being transitioned
            non_transition_particle = np.copy(particle)
            non_transition_weight = particle_weight * self.transition_noise

            # new particle for transitioned state
            particle[concepts_inconsistent] = 0
            particle /= np.sum(particle)
            transitioned_weight = particle_weight * (1 - self.transition_noise)

            # add new particles
            new_particles.append(non_transition_particle)
            new_particle_weights.append(non_transition_weight)
            new_particles.append(particle)
            new_particle_weights.append(transitioned_weight)

        return new_particle_weights, new_particles

    def assert_particle_limit(self):
        if len(self.particle_dists) > self.particle_num:
            while len(self.particle_dists) > self.particle_num:
                min_idx: int = np.argmin(self.particle_weights)
                del self.particle_dists[min_idx]
                del self.particle_weights[min_idx]

    def update_from_response(self, response, result):
        concepts_w_val = self.state_action_values[result[0]] == response

        for idx, particle in enumerate(self.particle_dists):
            current_weight = self.particle_weights[idx]

            # update weight of particle based on likelihood of producing the response
            p_z = np.sum(particle[concepts_w_val])
            new_weight = current_weight * ((1 - self.production_noise) * p_z + self.obs_noise_prob)

            self.particle_weights[idx] = new_weight

        self.check_particles_valid()

    def recreate_particles(self):
        self.particle_dists = []
        self.particle_weights = []
        particle1_dist = np.copy(self.prior)
        particle1_weight = 0.5
        self.particle_dists.append(particle1_dist)
        self.particle_weights.append(particle1_weight)

        # particle 2: consistent with observed data
        # TODO optimization: keep track of history particle instead of recalculating it; however, then history particle
        #  would be part of state
        particle2_dist = np.copy(self.prior)
        particle2_weight = 0.5

        for action_type, result, response in self.action_history:
            # update according to observed data
            particle2_dist = self.transition_model(particle2_dist, None, action_type, result, None)

        self.particle_dists.append(particle2_dist)
        self.particle_weights.append(particle2_weight)

        self.history_calcs += len(self.action_history)

    def observation_model(self, observation, new_state, action_type, action, concept_val):
        concepts_w_val = self.state_action_values[action[0]] == observation

        p_z = np.sum(new_state[concepts_w_val])

        return p_z

    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        concepts_inconsistent = self.state_action_values[action[0]] != action[1]
        new_state[concepts_inconsistent] = 0

        concept_prob_sum = np.sum(new_state)
        if concept_prob_sum == 0:
            # TODO check how it happens: iteration 10
            raise Exception("encountered degraded particle from history!")

        new_state /= concept_prob_sum

        return new_state

    def get_concept_prob(self, index):
        prob = 0

        for idx, particle in enumerate(self.particle_dists):
            prob += self.particle_weights[idx] * particle[index]

        return prob

    def get_observation_prob(self, action, observation):
        # TODO could be precomputed somewhere
        concepts_w_obs = self.state_action_values[action[0]] == observation

        prob = 0
        for idx, particle in enumerate(self.particle_dists):
            # TODO think about that it makes sense
            # TODO paper details
            consistent_prob = np.sum(particle[concepts_w_obs])
            response_prob_from_consistent = (1 - self.production_noise) * consistent_prob
            response_prob_from_inconsistent = self.obs_noise_prob

            prob += self.particle_weights[idx] * (response_prob_from_consistent + response_prob_from_inconsistent)

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
        self.init_particles()

        self.action_history = []

    def __copy__(self):
        new_model = ContinuousModel(self.prior, self.concept, particle_num=self.particle_num, verbose=self.verbose)
        new_model.set_state((self.particle_dists, self.particle_weights, self.action_history))

        return new_model
