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


ACTION_COSTS = {
    Actions.EXAMPLE: 7.0,
    Actions.QUIZ: 6.6,
    Actions.FEEDBACK: 12.0
}
