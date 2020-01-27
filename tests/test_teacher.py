import random
import numpy as np

from concepts.letter_addition import LetterAddition
from learner_models.continuous import ContinuousModel
from learner_models.memoryless import MemorylessModel
from teacher import Teacher


def test_class_creation():
    random.seed(123)
    np.random.seed(123)
    concept = LetterAddition(6)
    teacher = Teacher(concept)

    assert isinstance(teacher, Teacher)


def test_precompute():
    # profiling with 3 levels: 3:28m

    random.seed(123)
    np.random.seed(123)
    concept = LetterAddition(6)
    # Expected letters: {'A': 6, 'B': 3, 'C': 4, 'D': 0, 'E': 2, 'F': 5}
    teacher = Teacher(concept)
    teacher.setup(0)

    # initial belief uniformly distributed
    tree = {
        "belief": teacher.belief,
        "children": []
    }
    teacher.forward_plan(tree, 3, [45]*3)
    # print(tree)
    # one level tree
    assert len(tree["children"]) == 45

    expected_items = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
    ]
    for i in range(0, len(expected_items)):
        child = tree["children"][i*3]
        assert child['item'][0] == expected_items[i]


def test_planning2():
    random.seed(123)
    np.random.seed(123)
    concept = LetterAddition(6)
    # Expected letters: {'A': 6, 'B': 3, 'C': 4, 'D': 0, 'E': 2, 'F': 5}

    prior_distribution = np.array([1 / len(concept.get_concept_space()) for _ in range(len(concept.get_concept_space()))])
    model = MemorylessModel(prior_distribution, prior_distribution, concept)
    # model = ContinuousModel(prior_distribution, prior_distribution, concept)

    teacher = Teacher(concept, model, is_random=False)
    teacher.setup(0)

    # initial belief uniformly distributed
    tree = {
        "children": []
    }
    teacher.forward_plan(teacher.belief, tree, 2, [10]*2)

    print("")
    print(concept.get_true_concepts())
    # teacher.print_plan_tree(tree)

    print("Optimal path")
    print(teacher.find_optimal_action_path(tree))

