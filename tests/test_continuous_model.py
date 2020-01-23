from actions import Actions
from concepts.letter_addition import LetterAddition
from learner_models.memoryless import MemorylessModel

import numpy as np


def test_check_production_noise():
    number_range = list(range(6))
    concept = LetterAddition(6, number_range)
    state_space = concept.get_concept_space()

    prior_distribution = np.array([1 / len(state_space) for _ in range(len(state_space))])

    model = MemorylessModel(state_space, prior_distribution, concept)
    model.production_noise = 0

    belief = np.zeros_like(prior_distribution)
    for idx, new_state in enumerate(state_space):
        concept_val = concept.evaluate_concept(((0, 1), 1), new_state)
        p_z = model.observation_model(3, new_state, Actions.QUIZ, ((0, 1), 1), concept_val)

        p_s = 1

        belief[idx] = p_z * p_s

    assert np.count_nonzero(belief) == 0

    # 0-6
    # A+B=1|2|10|11: 240 matches; 2 pairs
    # A+B=3|4|8|9: 480 matches; 4 pairs
    # A+B=5|6|7: 720 matches; 6 pairs

    # 0-5
    # 1|2|8|9: 2 pairs: 48 matches
    # 3|4|6|7: 4 pairs: 96 matches
    # 5: 6 pairs: 144 matches

