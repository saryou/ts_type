import os
from pathlib import Path
from collections import defaultdict
from typing import Any, Type, Union, TypeVar, overload, Dict

from .nodes import TypeNode
from .builders import NodeBuilder


T = TypeVar('T')


class TypeDefinitionGenerator:
    """
    This class collects python objects which will be converted to
    typescript's types. And generates .ts files from collected objects
    into specified directory.
    """

    def __init__(self):
        self.types: Dict[str, Dict[str, Any]] = defaultdict(dict)

    @overload
    def add(self,
            target: T,
            output_filepath: str,
            output_name: str = ...) -> T:
        ...

    @overload
    def add(self, target: Type[T]) -> Type[T]:
        ...

    def add(self, target, output_filepath=None, output_name=None):
        if output_filepath is None:
            output_filepath = target.__module__.replace('.', '/')
            output_name = target.__name__
        else:
            assert output_filepath
            output_name = output_name or target.__name__
        _types = self.types[output_filepath]
        assert output_name not in _types,\
            f'{target} {output_filepath} {output_name}'
        _types[output_name] = target

        return target

    def generate(self,
                 output_dir: Union[str, Path],
                 builder_cls: Type[NodeBuilder] = NodeBuilder,
                 file_extension: str = 'gen.ts',
                 delete_conflicting_outputs: bool = True):
        basedir = Path(output_dir)
        file_extension = file_extension[1:]\
            if file_extension.startswith('.') else file_extension
        filepaths = {p for p in basedir.glob(f'**/*.{file_extension}')}\
            if delete_conflicting_outputs else set()

        output = self.render(builder_cls)
        for output_filepath in self.types.keys():
            filepath = basedir / (f'{output_filepath}.{file_extension}')
            filepaths.discard(filepath)

            _dirname = os.path.dirname(filepath)
            if not os.path.exists(_dirname):
                os.makedirs(_dirname)

            with open(filepath, 'w') as f:
                f.write(output[output_filepath])

        for filepath in filepaths:
            os.remove(filepath)

    def render(self,
               builder_cls: Type[NodeBuilder] = NodeBuilder) -> Dict[str, str]:
        result: Dict[str, str] = dict()
        for output_filepath, types in self.types.items():
            builder = builder_cls()
            defs: Dict[str, TypeNode] = dict()

            for name, value in types.items():
                if isinstance(value, TypeNode):
                    defs[name] = value
                else:
                    defs[name] = builder.type_to_node(value)

            builder.update_definitions(defs)
            result[output_filepath] = builder.render(set(defs.keys()))
        return result


__all__ = [
    'TypeDefinitionGenerator',
]
