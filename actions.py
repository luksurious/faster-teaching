from enum import Enum, auto


class Actions(Enum):
    EXAMPLE = auto()
    QUIZ = auto()
    QUESTION = auto()

    @staticmethod
    def all():
        return [Actions.EXAMPLE, Actions.QUIZ]  #, Actions.QUESTION]

    def __str__(self):
        return self._name_
