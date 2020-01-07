from learners.learner import SimLearner


def test_see_example():
    learner = SimLearner(6)

    pairs = learner.see_example(((0, 1), 6))

    print(pairs)