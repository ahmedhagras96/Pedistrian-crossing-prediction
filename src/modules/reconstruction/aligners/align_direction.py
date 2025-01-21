from enum import Enum


class AlignDirection(Enum):
    """
    Enumeration for specifying the direction of alignment.

    Attributes:
        LEFT (int): Align frames to the left of the key frame.
        RIGHT (int): Align frames to the right of the key frame.
        SPLIT (int): Align frames on both sides of the key frame.
    """
    LEFT = 1
    RIGHT = 2
    SPLIT = 3
