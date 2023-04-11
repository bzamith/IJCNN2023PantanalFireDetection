"""Module which contains the EnumClass"""

from enum import Enum


class EnumClass(Enum):
    """Enum base class"""

    @classmethod
    def is_member(cls, item: str) -> bool:
        """
        Verifies if value is a member of enum
        :param item: Value that wants to be checked if belongs to enum
        :return: Belongs (True) or not (False)
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True
