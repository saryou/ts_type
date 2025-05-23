import json
import re
from datetime import datetime, date, time
from enum import Enum
from contextlib import contextmanager
from dataclasses import fields as dc_fields, is_dataclass
from decimal import Decimal
from importlib import import_module
from typing import Optional, Any, Type, Callable, Union, ForwardRef, TypeVar, \
    Literal, List, Dict, Set, Tuple, cast, Generic
from typing import _TypedDictMeta  # type: ignore
from collections.abc import Sequence, MutableSequence, Mapping, \
    MutableMapping, Set as AbstractSet, MutableSet

from .exceptions import UnknownTypeError
from .utils import resolve_typevar
from . import nodes


T = TypeVar('T')
BuilderT = TypeVar('BuilderT', bound='NodeBuilder')

try:
    from types import UnionType  # type: ignore

    def _is_union_type(t: Any) -> bool:
        return isinstance(t, UnionType)
except ImportError:
    def _is_union_type(t: Any) -> bool:
        return False


try:
    from typing import Never  # type: ignore

    def _is_never_type(t: Any) -> bool:
        return t is Never
except ImportError:
    try:
        from typing_extensions import Never  # type: ignore

        def _is_never_type(t: Any) -> bool:
            return t is Never
    except ImportError:
        def _is_never_type(t: Any) -> bool:
            return False


try:
    from typing import TypeAliasType  # type: ignore

    def _resolve_type_alias_type(t: Any) -> Any:
        if isinstance(t, TypeAliasType):
            return _resolve_type_alias_type(t.__value__)
        else:
            return t
except ImportError:
    def _resolve_type_alias_type(t: Any) -> Any:
        return t


class NodeBuilder:
    """NodeBuilder converts python objects to typescript's types."""

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._definitions: Dict[str, nodes.TypeNode] = {}
        self._stack: List[Tuple[str, str, Any]] = []

    @property
    def definitions(self) -> Dict[str, Any]:
        return self._definitions

    def update_definitions(self, definitions: Dict[str, nodes.TypeNode]):
        self._definitions.update(definitions)

    def render(self, refs_to_export: Optional[Set[str]] = None) -> str:
        ctx = nodes.RenderContext(self.definitions)

        to_export = refs_to_export or set(self.definitions.keys())
        ref_names = [k for k in self.definitions if k in to_export]\
            + [k for k in self.definitions if k not in to_export]

        def render(k: str) -> str:
            export = 'export ' if k in to_export else ''

            ref = nodes.Reference(k)
            node = ctx.definitions[k]
            with ctx.enable_typevar_definition():
                lhs = f'{export}type {ref.render(ctx)}'
            rhs = f'{node.render(ctx)};'
            return f'{lhs} = {rhs}'

        return '\n\n'.join([render(k) for k in ref_names])

    def type_to_node(self, t: Any) -> nodes.TypeNode:
        t = _resolve_type_alias_type(t)

        if t in [None, type(None)]:
            return nodes.Null()

        origin = getattr(t, '__origin__', None)
        args: tuple[Any, ...] = getattr(t, '__args__', tuple())

        if origin is Union or _is_union_type(t):
            assert args
            if len(args) == 1:
                return self.type_to_node(args[0])
            return nodes.Union(
                of=[self.type_to_node(a) for a in args])
        elif origin is tuple:
            assert args
            return nodes.Tuple(
                [self.type_to_node(a) for a in args])
        elif origin in [list, set, Sequence, MutableSequence,
                        AbstractSet, MutableSet]:
            assert args
            return nodes.Array(self.type_to_node(args[0]))
        elif origin in [dict, Mapping, MutableMapping]:
            assert len(args) > 1
            key = self.type_to_node(args[0])
            assert isinstance(key, nodes.DictKeyType)
            return nodes.Dict(
                key=key,
                value=self.type_to_node(args[1]))
        elif origin is Literal:
            assert args
            literals = [self.literal_to_node(a) for a in args]
            if len(literals) > 1:
                return nodes.Union(of=cast(List[nodes.TypeNode], literals))
            return literals[0]
        elif origin:
            return self.handle_generic_type(t, origin, args)
        elif isinstance(t, str):
            return self.type_to_node(self._resolve_annotation(t))
        elif isinstance(t, TypeVar):
            try:
                _t = self._resolve_typevar(t)
            except AssertionError:
                return nodes.TypeVariable(typevar=t)
            return self.type_to_node(_t)
        elif isinstance(t, ForwardRef):
            _globals = self._current_module.__dict__
            try:
                evaluated = t._evaluate(_globals, None, set(),  # type: ignore
                                        recursive_guard=set())  # type: ignore
            except TypeError:
                evaluated = t._evaluate(_globals, None, set())  # type: ignore
            return self.type_to_node(evaluated)
        elif isinstance(t, NodeCompatible):
            return t.convert_to_node(self)
        elif isinstance(t, type):
            if issubclass(t, NodeCompatibleClass):
                return t.convert_to_node(self)
            elif issubclass(t, (str, datetime, date, time)):
                return nodes.String()
            elif issubclass(t, bool):
                return nodes.Boolean()
            elif issubclass(t, (int, float, Decimal)):
                return nodes.Number()
            elif issubclass(t, Enum):
                return self.enum_to_node(t)
            elif is_dataclass(t):
                return self.dataclass_to_node(t)
            elif isinstance(t, _TypedDictMeta):
                return self.typeddict_to_node(t)
        elif _is_never_type(t):
            return nodes.Never()

        return self.handle_unknown_type(t)

    def handle_generic_type(self,
                            t: Any,
                            origin: type,
                            args: tuple[Any, ...]) -> nodes.TypeNode:
        node = self.type_to_node(origin)
        if isinstance(node, nodes.Reference):
            node.typevars = [self.type_to_node(t) for t in args]
        return node

    def handle_unknown_type(self, t: Any) -> nodes.TypeNode:
        raise UnknownTypeError(f'Type `{t}` is not supported.')

    class RefSource:
        def __init__(self,
                     type: Union[type, None],
                     identifier: str):
            self.type = type
            self.identifier = identifier

        @classmethod
        def from_type(cls, t: type) -> 'NodeBuilder.RefSource':
            return cls(
                type=t,
                identifier=re.sub(r'\.', '__', '.'.join(
                    [t.__module__, t.__qualname__]))
            )

        @classmethod
        def from_identifier(cls, t: str) -> 'NodeBuilder.RefSource':
            return cls(type=None, identifier=t)

    def define_ref_node(
            self,
            source: Union[RefSource, type, str],
            define: Callable[[], nodes.TypeNode],
            generic_params: Optional[Sequence[
                Union[TypeVar, nodes.TypeVariable]]] = None,
            ref_typevars: List[nodes.TypeNode] = []) -> nodes.Reference:
        if isinstance(source, str):
            source = self.RefSource.from_identifier(source)
        elif isinstance(source, type):
            source = self.RefSource.from_type(source)

        t = source.type
        _id = source.identifier

        defs = self._definitions
        if _id not in defs:
            defs[_id] = nodes.TypeNode()

            try:
                with self._begin_module_context(t):
                    type_node = define()
                    if generic_params is not None:
                        type_node.add_generic_params(
                            [p if isinstance(p, nodes.TypeVariable)
                             else nodes.TypeVariable(p)
                             for p in generic_params])
                    elif (params := getattr(t, '__parameters__', None)):
                        type_node.add_generic_params(
                            [nodes.TypeVariable(p) for p in params])
                defs[_id] = type_node
            except Exception:
                del defs[_id]
                raise

        return nodes.Reference(_id, ref_typevars)

    def literal_to_node(self, value: Any) -> nodes.Literal:
        assert isinstance(value, (int, bool, str))
        return nodes.Literal(json.dumps(value))

    def enum_to_node(self, enum: Type[Enum]) -> nodes.TypeNode:
        return self.define_ref_node(enum, lambda: nodes.Union(
            [self.literal_to_node(i.name) for i in enum]))

    def dataclass_to_node(self, dc: Type[Any]) -> nodes.TypeNode:
        return self.define_ref_node(
            dc,
            lambda: nodes.Object(
                attrs={f.name: self.type_to_node(f.type)
                       for f in dc_fields(dc)},
                omissible=set()))

    def typeddict_to_node(self, t: _TypedDictMeta) -> nodes.TypeNode:
        return self.define_ref_node(t, lambda: nodes.Object(
            attrs={k: self.type_to_node(v)
                   for k, v in t.__annotations__.items()},
            omissible=t.__optional_keys__))

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


class NodeCompatibleClass(Generic[BuilderT]):
    @classmethod
    def convert_to_node(cls, builder: BuilderT) -> nodes.TypeNode:
        raise NotImplementedError()


class NodeCompatible(Generic[BuilderT]):
    def convert_to_node(self, builder: BuilderT) -> nodes.TypeNode:
        raise NotImplementedError()


__all__ = [
    'NodeBuilder',
    'NodeCompatibleClass',
    'NodeCompatible',
]
