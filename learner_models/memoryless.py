from learner_models.base_belief import BaseBelief
from concepts.concept_base import ConceptBase
from actions import Actions

import numpy as np


# TODO check why as 4th action a quiz is chosen
class MemorylessModel(BaseBelief):

    def __init__(self, belief_state, prior, concept: ConceptBase, trans_noise=0.15, prod_noise=0.019,
                 verbose: bool = True):
        super().__init__(belief_state, prior, concept, trans_noise=trans_noise, prod_noise=prod_noise, verbose=verbose)

    def belief_update_formula(self, action_type, action, observation):
        """
        Override explicit loop version for more efficient calculations
        """
        new_belief = self.belief_state.copy()

        constrain_belief = observation is not None and action[1] is None
        if constrain_belief:
            self.obs_update(action, new_belief, observation)

        # evidence is given
        transition_happened = action[1] is not None and action[1] != observation
        if transition_happened:
            self.trans_update(action, new_belief)

        return new_belief

    def obs_update(self, action, new_belief, observation):
        # Quiz/Feedback action
        # belief can be sharpened
        consistent_states = self.state_action_values[action[0]] == observation

        # prob of consistent concepts with observation and action --> 1-e + random result prob
        prob_consistent = (1 - self.production_noise) + self.obs_noise_prob
        new_belief[consistent_states] = self.belief_state[consistent_states] * prob_consistent

        # prob of inconsistent concepts with observation and action --> e
        new_belief[~consistent_states] = self.belief_state[~consistent_states] * self.obs_noise_prob

    def trans_update(self, action, new_belief):
        # state might have changed
        consistent_states = self.find_consistent_states_for_transition(action)

        # TODO assuming uniform prior
        uniform_cons_trans_prob = 1 / np.count_nonzero(consistent_states)
        incons_belief_prob = np.sum(self.belief_state[~consistent_states])
        transition_prob = uniform_cons_trans_prob * (1 - self.transition_noise) * incons_belief_prob

        new_belief[~consistent_states] = self.belief_state[~consistent_states] * self.transition_noise
        new_belief[consistent_states] = self.belief_state[consistent_states] + transition_prob

    def find_consistent_states_for_transition(self, action):
        consistent_states = self.state_action_values[action[0]] == action[1]
        return consistent_states

    # "Raw" implementations of the bayesian belief update models; more costly because of looping
    def observation_model(self, observation, new_state, action_type, action, concept_val):
        """
        Probability of seeing observation (i.e. response of the learner) in the (new) state given the taken action
        """
        # example action: there is no observation
        if observation is None:
            return 1

        # quiz/question action: the observation is deterministic based on if it matches the state of the learner
        # Note: single belief update for question w/ feedback action not possible!
        if concept_val == observation:
            return 1 - self.production_noise + self.obs_noise_prob

        return self.obs_noise_prob

    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        """
        Probability of going to new state given an action (and current state)
        """

        if action_type == Actions.QUIZ:
            # no state change expected - but we can rule out states that do not match her response
            # no need for a loop
            return self.belief_state[new_idx]  # transition prob only to same state is 1, only b(s) left in formula

        p_s = 0

        if concept_val == action[1]:
            p_s = self.calculate_ps(action, new_idx)

        return p_s

    def calculate_ps(self, action, new_idx):
        b_s = self.belief_state

        consistent_state_filter = self.find_consistent_states_for_transition(action)

        # prob of going to new state
        noisy_prior = self.prior[new_idx] / np.sum(self.prior[consistent_state_filter]) - self.transition_noise

        p_s = np.ones(len(self.hypotheses)) * noisy_prior  # default transition with prior probability
        p_s[consistent_state_filter] = 0  # no transition from other consistent concepts

        p_s[new_idx] = 1  # instead it stays at the same concept
        p_s = np.sum(p_s * b_s)

        return p_s

    def see_action(self, action_type, action):
        pass

    def get_observation_prob(self, action, observation):
        concepts_w_obs = self.state_action_values[action[0]] == observation
        return np.sum(self.belief_state[concepts_w_obs])

    def __copy__(self):
        return MemorylessModel(self.belief_state.copy(), self.prior, self.concept, trans_noise=self.transition_noise,
                               prod_noise=self.production_noise, verbose=self.verbose)
