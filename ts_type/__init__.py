import json
import re
import os
from pathlib import Path
from datetime import datetime, date, time
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import fields as dc_fields, is_dataclass
from importlib import import_module
from types import GenericAlias
from typing import Optional, Any, Type, Callable, Union, ForwardRef, TypeVar,\
    overload, Literal, List, Dict, Set, Tuple, cast, Generic

from .exceptions import UnknownTypeError
from .nodes import ArrayNode, BooleanNode, DictKeyType, DictNode,\
    LiteralNode, NullNode, NumberNode, ObjectNode, ReferenceNode,\
    RenderContext, StringNode, TupleNode, TypeNode, TypeVariableNode,\
    UnionNode, UnknownNode
from .nodes import *  # noqa


T = TypeVar('T')
BuilderT = TypeVar('BuilderT', bound='NodeBuilder')


class NodeBuilder:
    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._definitions: Dict[str, TypeNode] = {}
        self._stack: List[Tuple[str, str, Any]] = []

    @property
    def definitions(self) -> Dict[str, Any]:
        return self._definitions

    def update_definitions(self, definitions: Dict[str, TypeNode]):
        self._definitions.update(definitions)

    def render(self, refs_to_export: Optional[Set[str]] = None) -> str:
        ctx = RenderContext(self.definitions)

        to_export = refs_to_export or set(self.definitions.keys())
        ref_names = [k for k in self.definitions if k in to_export]\
            + [k for k in self.definitions if k not in to_export]

        def render(k: str) -> str:
            export = 'export ' if k in to_export else ''
            ref = ReferenceNode(k)
            node = ctx.definitions[k]
            lhs = f'{export}type {ref.render(ctx)}'
            rhs = f'{node.render(ctx)};'
            return f'{lhs} = {rhs}'

        return '\n\n'.join([render(k) for k in ref_names])

    def type_to_node(self, t: Any, unknown_node: bool = False) -> TypeNode:
        if t in [None, type(None)]:
            return NullNode()

        origin = getattr(t, '__origin__', None)
        args = getattr(t, '__args__', tuple())

        if origin is Union:
            assert args
            if len(args) == 1:
                return self.type_to_node(args[0], unknown_node)
            return UnionNode(
                of=[self.type_to_node(a, unknown_node) for a in args])
        elif origin is tuple:
            assert args
            return TupleNode(
                [self.type_to_node(a, unknown_node) for a in args])
        elif origin is list or origin is set:
            assert args
            return ArrayNode(self.type_to_node(args[0], unknown_node))
        elif origin is dict:
            assert len(args) > 1
            key = self.type_to_node(args[0], unknown_node)
            assert isinstance(key, DictKeyType)
            return DictNode(
                key=key,
                value=self.type_to_node(args[1], unknown_node))
        elif origin is Literal:
            assert args
            literals = [self._literal_to_node(a) for a in args]
            if len(literals) > 1:
                return UnionNode(of=cast(List[TypeNode], literals))
            return literals[0]
        elif origin:
            node = self.type_to_node(origin, unknown_node)
            if isinstance(node, ReferenceNode):
                node.typevars = [self.type_to_node(t) for t in args]
            return node
        elif isinstance(t, str):
            return self.type_to_node(self._resolve_annotation(t), unknown_node)
        elif isinstance(t, TypeVar):
            try:
                _t = self._resolve_typevar(t)
            except AssertionError:
                return TypeVariableNode(typevar=t)
            return self.type_to_node(_t, unknown_node)
        elif isinstance(t, ForwardRef):
            _globals = self._current_module.__dict__
            evaluated = t._evaluate(_globals, None, set())  # type: ignore
            return self.type_to_node(evaluated, unknown_node)
        elif isinstance(t, NodeCompatible):
            return t.convert_to_node(self)
        elif isinstance(t, type):
            if issubclass(t, NodeCompatibleClass):
                return t.convert_to_node(self)
            elif issubclass(t, (str, datetime, date, time)):
                return StringNode()
            elif issubclass(t, bool):
                return BooleanNode()
            elif issubclass(t, (int, float)):
                return NumberNode()
            elif issubclass(t, Enum):
                return self.define_ref_node(t, lambda: UnionNode(
                    [self._literal_to_node(i) for i in t]))
            elif is_dataclass(t):
                return self.define_ref_node(
                    t,
                    lambda: ObjectNode(
                        attrs={f.name: self.type_to_node(f.type, unknown_node)
                               for f in dc_fields(t)},
                        omissible=set()))

        return self.handle_unknown_type(t, unknown_node)

    def handle_unknown_type(self,
                            t: Any,
                            unknown_node: bool) -> TypeNode:
        if unknown_node:
            return UnknownNode()
        raise UnknownTypeError(f'Type `{t}` is not supported.')

    def define_ref_node(self,
                        type_or_id: Union[type, str],
                        define: Callable[[], TypeNode],
                        generic_params: Optional[List[TypeVar]] = None,
                        ref_typevars: List[TypeNode] = []) -> ReferenceNode:
        if isinstance(type_or_id, str):
            t = None
            _id = type_or_id
        else:
            t = type_or_id
            _id = '.'.join([t.__module__, t.__qualname__])
            _id = re.sub(r'\.', '__', _id)

        defs = self._definitions
        if _id not in defs:
            defs[_id] = TypeNode()

            try:
                with self._begin_module_context(t):
                    type_node = define()
                    if generic_params is not None:
                        type_node.add_generic_params(
                            [TypeVariableNode(p) for p in generic_params])
                    elif (params := getattr(t, '__parameters__', None)):
                        type_node.add_generic_params(
                            [TypeVariableNode(p) for p in params])
                defs[_id] = type_node
            except Exception:
                del defs[_id]
                raise

        return ReferenceNode(_id, ref_typevars)

    def _literal_to_node(self, value: Any) -> LiteralNode:
        literal = value.name if isinstance(value, Enum) else value
        assert isinstance(literal, (int, bool, str))
        return LiteralNode(json.dumps(literal))

    @contextmanager
    def _begin_module_context(self, t: Optional[Type]):
        if t is None:
            yield
            return

        self._import_module(t.__module__)
        self._stack.append((t.__module__, t.__qualname__, t))
        yield
        self._stack.pop()

    @property
    def _current_module(self) -> Any:
        return self._modules[self._stack[-1][0]]

    @property
    def _current_obj(self) -> Any:
        return self._stack[-1][-1]

    def _import_module(self, module):
        if module not in self._modules:
            self._modules[module] = import_module(module)

    def _resolve_typevar(self, t: Any) -> Optional[Type]:
        return resolve_typevar(self._current_obj, t)

    def _resolve_annotation(self, qualname: str) -> Type:
        name, *names = qualname.split('.')
        t = getattr(self._current_module, name)
        for name in names:
            t = getattr(t, name)
        return t

    def __str__(self) -> str:
        return self.render(None)


def resolve_typevar(cls: type, t: TypeVar) -> type:
    return _resolve_typevar(cls, t, 0)


def _resolve_typevar(
        cls: type,
        t: TypeVar,
        mro_index: int) -> Any:
    if isinstance(t, GenericAlias):
        ev_args = tuple(_resolve_typevar(cls, a, mro_index)
                        for a in t.__args__)
        if ev_args == t.__args__:
            return t
        return GenericAlias(t.__origin__, ev_args)

    if not isinstance(t, TypeVar):
        return t

    mro = list(reversed(cls.mro()))
    arg_index: Optional[int] = None
    found = _find_typevar(mro[mro_index:], t)
    if found:
        mro_index = mro_index + found[0] + 1
        arg_index = found[1]

        for m in mro[mro_index:]:
            for b in getattr(m, '__orig_bases__', []):
                args = getattr(b, '__args__', [])
                if args and len(args) > arg_index:
                    arg = args[arg_index]
                    assert t not in getattr(arg, '__parameters__', []),\
                        'Unsupported usage of generic parameters. '\
                        f'`{t}` has represented other generic type which '\
                        f'contains `{T}` itself (`{arg}` in `{m}`). '\
                        'So we can not determine which use is '\
                        'appropriate in this context. You should substitute '\
                        'generic type parameter which has different name '\
                        f'such as `OtherNameT` for `{t}`.'
                    concrete_type = _resolve_typevar(cls, arg, mro_index)
                    return concrete_type

    raise AssertionError(
        f'Can not resolve TypeVar `{t}` for '
        f'({cls.__module__}.{cls.__qualname__})')


def _find_typevar(mro: List[type], t: TypeVar) -> Optional[Tuple[int, int]]:
    for i, m in enumerate(mro):
        for base in getattr(m, '__orig_bases__', []):
            for _i, p in enumerate(getattr(base, '__parameters__', [])):
                if p == t:
                    return (i, _i)
    return None


class NodeCompatibleClass(Generic[BuilderT]):
    @classmethod
    def convert_to_node(cls, builder: BuilderT) -> TypeNode:
        raise NotImplementedError()


class NodeCompatible(Generic[BuilderT]):
    def convert_to_node(self, builder: BuilderT) -> TypeNode:
        raise NotImplementedError()


class TypeDefinitionGenerator:
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


generator = TypeDefinitionGenerator()


def gen_type(t: Type[T]) -> Type[T]:
    return generator.add(t)
