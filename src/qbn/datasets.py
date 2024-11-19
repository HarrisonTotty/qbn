from __future__ import annotations

import itertools
import random

from keras.datasets import mnist

from qbn.bf import BooleanArray
from qbn.training import ClassificationMap

__all__ = ["get_left_vs_right", "get_mnist"]

def _left_right_both(val: list[bool]) -> str:
    any_left = any(v for v in val[:len(val) // 2])
    any_right = any(v for v in val[len(val) // 2:])
    if any_left and not any_right:
        return 'left'
    elif not any_left and any_right:
        return 'right'
    elif any_left and any_right:
        return 'both'
    return 'neither'

def get_left_vs_right(length: int = 16) -> tuple[ClassificationMap, ClassificationMap]:
    """
    Obtains a copy of LVR data of the specified length.
    """
    combinations = [list(v) for v in itertools.product([True, False], repeat=length)]
    random.shuffle(combinations)
    training = ClassificationMap()
    test = ClassificationMap()
    for sample in combinations[:len(combinations) // 2]:
        training[BooleanArray(sample)] = _left_right_both(sample)
    for sample in combinations[0:len(combinations) // 2:]:
        test[BooleanArray(sample)] = _left_right_both(sample)
    return (training, test)


def get_mnist(true_threshold: int = 64) -> tuple[ClassificationMap, ClassificationMap]:
    """
    Obtains the MNIST dataset in `ClassificationMap` format.
    """
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    training = ClassificationMap()
    test = ClassificationMap()
    for i, sample in enumerate(train_x[0:1000]):
        digit = str(train_y[i])
        value = [v > true_threshold for v in sample.flatten().tolist()]
        training[BooleanArray(value)] = digit
    for i, sample in enumerate(test_x[0:1000]):
        digit = str(test_y[i])
        value = [v > true_threshold for v in sample.flatten().tolist()]
        test[BooleanArray(value)] = digit
    return (training, test)
