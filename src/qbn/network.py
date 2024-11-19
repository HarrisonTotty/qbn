"""
Contains the definition of a network.
"""
from __future__ import annotations

import statistics

import pickle
from collections import UserList
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from random import randint
from typing import Optional

from bloqade.atom_arrangement import ListOfLocations

from qbn.bf import BooleanArray, BooleanFunction, random_function
from qbn.training import AccuracyResults, ClassificationMap, TrainingResults

class Layer(UserList[BooleanFunction]):
    """
    Represents a layer of the boolean network.
    """
    data: list[BooleanFunction]

    def __call__(self, val: list[bool]) -> list[bool]:
        """
        Applies the layer to the specified list of boolean values.
        """
        result = []
        for i, v in enumerate(val):
            i2 = i + 1
            v2 = val[i2] if i2 < len(val) else val[0]
            f = self[i] if i < len(self) else self[i % len(self)]
            result.append(f(v, v2))
        return result

    def __repr__(self) -> str:
        """
        Gets the string representation of the layer.
        """
        return str([f.n for f in self])

    @staticmethod
    def build(funcs: list[int]) -> Layer:
        """
        Builds a new layer from the specified list of boolean function codes.
        """
        return Layer([BooleanFunction(i) for i in funcs])

class LayerList(UserList[Layer]):
    """
    Represents a list of layers.
    """
    data: list[Layer]

    def __call__(self, val: list[bool]) -> list[bool]:
        """
        Applies this layer list to the specified list of boolean values.
        """
        return reduce(lambda val, layer: layer(val), self, val)

    def __repr__(self) -> str:
        """
        Gets the string representation of the layer list.
        """
        return '\n'.join([layer.__repr__() for layer in self])

    @staticmethod
    def build(funcs: list[list[int]]) -> LayerList:
        """
        Builds a layer list from a list of boolean function integers.
        """
        return LayerList([Layer.build(layer) for layer in funcs])

    def accuracy(self, data: ClassificationMap) -> AccuracyResults:
        """
        Determines the accuracy of the network according to the specified data.
        """
        # First, we evaluate all of our training data and store the results
        # organized by intended classification.
        results: dict[str, list[BooleanArray]] = {}
        for ival, iclass in data.items():
            oval = BooleanArray(self(ival.data))
            if iclass in results:
                results[iclass].append(oval)
            else:
                results[iclass] = [oval]
        # Now, we find the most common answer for each category and calculate
        # what proportion of answers in each category match their most common
        # answer. We also keep track of what the most common answer was for
        # building a classification map, and whether the system has unique
        # solutions.
        valid: bool = True
        classifier: dict[BooleanArray, str] = {}
        class_accuracies: dict[str, float] = {}
        for rclass, rvals in results.items():
            most = statistics.mode(rvals)
            # If this answer already exists in the classifier, we have an
            # invalid network and need to report both accuracies as 0.
            if most in classifier:
                valid = False
                other_class = classifier[most]
                class_accuracies[other_class] = 0.0
                class_accuracies[rclass] = 0.0
            else:
                classifier[most] = rclass
                class_accuracies[rclass] = round(rvals.count(most) / len(rvals), 2)
        # We convert all of the information we learned above into an accuracy
        # results object.
        return AccuracyResults(
            accuracy = round(statistics.mean(class_accuracies.values()), 2),
            class_accuracies = class_accuracies,
            classifications = ClassificationMap(classifier),
            valid = valid
        )

@dataclass
class Network:
    """
    Represents a boolean network.
    """
    layers: LayerList
    classifier: Optional[ClassificationMap] = None

    def __call__(self, val: list[bool]) -> str | None:
        """
        Applies the network to the specified list of values.
        """
        return self.classify(val)

    def accuracy(self, data: ClassificationMap) -> AccuracyResults:
        """
        Determines the accuracy of the network according to the specified data.
        """
        return self.layers.accuracy(data)

    @staticmethod
    def build(layer_sizes: list[int]) -> Network:
        """
        Builds a new random network from the specified layer sizes.
        """
        funcs = []
        for size in layer_sizes:
            funcs.append([random_function() for _ in range(size)])
        return Network(
            layers = LayerList.build(funcs),
        )

    def classify(self, val: list[bool]) -> str | None:
        """
        Classifies the specified input based on the built-in classifier.
        """
        if self.classifier is None:
            raise RuntimeError('please train the network first!')
        return self.classifier.get(BooleanArray(self.evaluate(val)))

    def evaluate(self, val: list[bool]) -> list[bool]:
        """
        Evaluates the network against the specified list of values.
        """
        return self.layers(val)

    def atom_arrangement(self) -> ListOfLocations:
        """
        Gets the list of atoms that would make up this cluster.
        """


    @staticmethod
    def load(path: Path) -> Network:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def is_trained(self) -> bool:
        return self.classifier is None

    def save(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def train_classical(self, data: ClassificationMap, goal: float = 0.95, changes_per_iter: int = 1, max_iterations: int = 10000) -> TrainingResults:
        """
        Trains the network according to the specified training data.
        """
        current_accuracy = self.layers.accuracy(data)
        results: list[AccuracyResults] = []
        for i in range(max_iterations):
            results.append(current_accuracy)
            if current_accuracy.accuracy >= goal:
                break
            layers = deepcopy(self.layers)
            for _ in range(changes_per_iter):
                i = randint(0, len(layers) - 1)
                j = randint(0, len(layers[i]) - 1)
                layers[i][j] = BooleanFunction.random()
            new_accuracy = layers.accuracy(data)
            if new_accuracy.valid and new_accuracy.accuracy > current_accuracy.accuracy:
                current_accuracy = new_accuracy
                self.layers = layers
                self.classifier = current_accuracy.classifications
                print(f'New Accuracy: {current_accuracy.accuracy}')
        return TrainingResults(
            accuracy = current_accuracy.accuracy,
            class_accuracies = current_accuracy.class_accuracies,
            classifications = current_accuracy.classifications,
            valid = current_accuracy.valid,
            results = results
        )

    def train_quantum(self, data: ClassificationMap, goal: float = 0.95) -> TrainingResults:
        """
        Trains the network using quantum computations.
        """
