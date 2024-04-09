from enum import Enum
from fastapi import HTTPException

from typing import Optional, List

class Domain(Enum):
    HOUSE = "house"
    TREE = "tree"
    PERSON = "person"


    @classmethod
    def from_str(cls, value: str) -> Optional["Domain"]:
        for member in cls:
            if member.value == value:
                return member

        raise HTTPException(400, "Invalid domain")

    @classmethod
    def get_value_list(cls) -> List[str]:
        return [cls.HOUSE.value, cls.TREE.value, cls.PERSON.value]


class DBDomain(Enum):
    house = "house"
    tree = "tree"
    person = "person"



