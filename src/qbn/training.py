'''
Contains functions associated with training boolean functions.
'''
from __future__ import annotations
from collections import UserDict
from dataclasses import dataclass

from qbn.bf import BooleanArray

__all__ = ["AccuracyResults", "ClassificationMap", "TrainingResults"]

class ClassificationMap(UserDict[BooleanArray, str]):
    """
    A mapping from an array of boolean values to a corresponding string value.
    """
    data: dict[BooleanArray, str]

    def __repr__(self) -> str:
        return '{' + ', '.join(f'{k} -> {v}' for k, v in self.items()) + '}'

    def __str__(self) -> str:
        return self.__repr__()

@dataclass
class AccuracyResults:
    """
    Represents results obtained from the accuracy determination.
    """
    accuracy: float
    class_accuracies: dict[str, float]
    classifications: ClassificationMap
    valid: bool

    def __repr__(self) -> str:
        return f"""
Valid?:            {self.valid}
Overall Accuracy:  {self.accuracy}
Accuracy By Class: {self.class_accuracies}

Classifications:
{self.classifications}
        """.strip()

@dataclass
class TrainingResults(AccuracyResults):
    """
    Represents the results on training the network.
    """
    results: list[AccuracyResults]

    def __repr__(self) -> str:
        return f"""
Valid?:            {self.valid}
Overall Accuracy:  {self.accuracy}
Accuracy By Class: {self.class_accuracies}

Classifications:
{self.classifications}
        """.strip()
