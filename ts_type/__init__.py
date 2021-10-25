from typing import Type, TypeVar

from .generators import TypeDefinitionGenerator

from .builders import *  # noqa
from .exceptions import *  # noqa
from .generators import *  # noqa
from .nodes import *  # noqa
from .utils import *  # noqa
from .version import VERSION  # noqa


T = TypeVar('T')


generator = TypeDefinitionGenerator()


def gen_type(t: Type[T]) -> Type[T]:
    return generator.add(t)
