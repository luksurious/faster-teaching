import time
import numpy as np

from actions import Actions
from concepts.concept_base import ConceptBase
from learner_models.base_belief import BaseBelief
from learner_models.continuous import ContinuousModel
from planners.base_planner import BasePlanner


class MaxInformationGainPlanner(BasePlanner):
    def __init__(self, concept: ConceptBase, actions: list, belief: BaseBelief, verbose: bool = False):
        super().__init__(concept, actions)

        assert isinstance(belief, ContinuousModel), "Max Info Gain only supports the continuous model"

        self.belief = belief

        self.concept_space = self.concept.get_concept_space()

        self.verbose = verbose

    def perform_preplanning(self, a=None, b=None):
        pass

    def load_preplanning(self, data):
        pass

    def start_teaching_phase(self):
        pass

    def reset(self):
        pass

    def choose_action(self, prev_response=None):
        start_time = time.time()

        action_data = self.find_max_gain_item(self.belief.copy())

        plan_duration = time.time() - start_time
        self.plan_duration_history.append(plan_duration)
        if self.verbose:
            print("// planning took %.2f" % plan_duration)

        return action_data

    def find_max_gain_item(self, belief: ContinuousModel):
        samples = self.concept.get_rl_actions()

        # save state to reset to later
        model_state = belief.get_state()

        # max_gain = -1
        gains = []
        # best_action = None
        actions = []

        entropy_before = self.calc_entropy(belief)

        for item in samples:
            value = self.concept.evaluate_concept([item])
            result = (item, value)

            for teaching_action in self.actions:

                cur_gain = self.calc_information_gain(belief, result, teaching_action, entropy_before)
                belief.set_state(model_state)

                # if cur_gain > max_gain:
                #     max_gain = cur_gain
                #     best_action = (teaching_action,) + result
                gains.append(cur_gain)
                actions.append((teaching_action,) + result)

        gains = np.array(gains)
        max_idxs = np.flatnonzero(gains.max() == gains)

        return actions[np.random.choice(max_idxs)]

    def calc_information_gain(self, belief: ContinuousModel, result, teaching_action, entropy_before):
        gain = 0

        if teaching_action == Actions.EXAMPLE:
            # no observations
            expected_obs = None

            belief.update_belief(teaching_action, result, expected_obs)

            entropy_after = self.calc_entropy(belief)

            gain = entropy_before - entropy_after

        return gain

    @staticmethod
    def calc_entropy(belief: ContinuousModel):
        total_entropy = 0

        for idx, particle in enumerate(belief.particle_dists):
            non_zero_items = particle[particle > 0]
            entropy = -np.sum(non_zero_items * np.log(non_zero_items))

            total_entropy += belief.particle_weights[idx] * entropy

        return total_entropy
