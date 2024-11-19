"""
Contains definitions associated with boolean functions.
"""
from __future__ import annotations
from collections import UserList
from random import randint
from typing import Callable

__all__ = ["BooleanArray", "BooleanFunction"]

def nand(a: bool, b: bool) -> bool:
    return not (a and b)

def nor(a: bool, b: bool) -> bool:
    return not (a or b)

def xnor(a: bool, b: bool) -> bool:
    return (a and b) or (not a and not b)

def xor(a: bool, b: bool) -> bool:
    return (a and not b) or (not a and b)

FUNCS: dict[int, Callable[[bool, bool], bool]] = {
    0: lambda a, b: False,
    1: lambda a, b: a and b,
    2: lambda a, b: a and not b,
    3: lambda a, _: a,
    4: lambda a, b: not a and b,
    5: lambda _, b: b,
    6: xor,
    7: lambda a, b: a or b,
    8: nor,
    9: xnor,
    10: lambda _, b: not b,
    11: lambda a, b: a or not b,
    12: lambda a, _: not a,
    13: lambda a, b: not a or b,
    14: nand,
    15: lambda a, b: True,
}

REPR: dict[int, str] = {
    0: "FALSE",
    1: "AND",
    2: "AND NOT",
    3: "A",
    4: "NOT AND",
    5: "B",
    6: "XOR",
    7: "OR",
    8: "NOR",
    9: "XNOR",
    10: "NOT B",
    11: "OR NOT",
    12: "NOT A",
    13: "NOT OR",
    14: "NAND",
    15: "TRUE",
}

class BooleanArray(UserList[bool]):
    """
    Represents a hashable array of boolean values.
    """
    data: list[bool]

    def __hash__(self) -> int:
        return sum(b << i for i, b in enumerate(self.data))

    def __repr__(self) -> str:
        return ''.join('1' if x else '0' for x in self)

    def __str__(self) -> str:
        return self.__repr__()

def random_function(exclude_0_15: bool = True, exclude_unary: bool = True) -> int:
    """
    Generates a random boolean function number.
    """
    res = randint(1, 14) if exclude_0_15 else randint(0, 15)
    if exclude_unary and res in [3, 5, 10, 12]:
        return random_function(exclude_0_15, exclude_unary)
    return res

class BooleanFunction:
    """
    Represents a boolean function of two variables.
    """
    n: int
    func: Callable[[bool, bool], bool]

    def __getstate__(self) -> int:
        return self.n

    def __setstate__(self, n: int) -> None:
        self.n = n
        self.func = FUNCS[n]

    def __init__(self, n: int) -> None:
        if n < 0 or n > 15:
            raise ValueError('Value for n must be between 0 and 15.')
        self.n = n
        self.func = FUNCS[n]

    def __repr__(self) -> str:
        return REPR[self.n]

    def __call__(self, a: bool, b: bool) -> bool:
        return self.func(a, b)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BooleanFunction):
            raise ValueError('can only compare boolean functions!')
        return self.n == value.n

    @staticmethod
    def random(exclude_0_15: bool = True, exclude_unary: bool = True) -> BooleanFunction:
        """
        Generates a random boolean function.
        """
        return BooleanFunction(random_function(exclude_0_15, exclude_unary))
