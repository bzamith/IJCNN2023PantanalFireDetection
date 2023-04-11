"""Module with utilities for validating the user inputs"""
from enum import Enum
from typing import Any, Type


def validate_enum(input_value: Any, input_name: str, enum_class: Type[Enum]) -> bool:
    """
    Validate if an input belongs to enum class
    :param input_value: the input value
    :param input_name: the input name
    :param enum_class: the enum class
    :return: True if belongs, else raise exception
    """
    if not enum_class.is_member(input_value):
        raise TypeError("Parameter {} must belong to {}".format(input_name, enum_class.__name__))
    return True


def validate_type(input_value: Any, input_name: str, type_class: Any) -> bool:
    """
    Validate if an input belongs to a given type or class
    :param input_value: the input value
    :param input_name: the input name
    :param type_class: the class or type
    :return: True if belongs, else raise exception
    """
    if not isinstance(input_value, type_class):
        raise TypeError("Parameter {} must be from type {}".format(input_name, type_class.__name__))
    return True
