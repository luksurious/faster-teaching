from enum import Enum, auto


class Actions(Enum):
    EXAMPLE = auto()
    QUIZ = auto()
    FEEDBACK = auto()

    @staticmethod
    def all():
        return [Actions.EXAMPLE, Actions.QUIZ, Actions.FEEDBACK]

    @staticmethod
    def qe_only():
        return [Actions.EXAMPLE, Actions.QUIZ]

    def __str__(self):
        return self._name_


ACTION_COSTS_LETTERS = {
    Actions.EXAMPLE: 7.0,
    Actions.QUIZ: 6.6,
    Actions.FEEDBACK: 12.0
}
ACTION_COSTS_NUMBER = {
    Actions.EXAMPLE: 2.4,
    Actions.QUIZ: 2.8,
    Actions.FEEDBACK: 4.8
}
ACTION_COSTS_SAMPLE = {
    Actions.EXAMPLE: 0.0,
    Actions.QUIZ: 0.0,
    Actions.FEEDBACK: 0.0
}
