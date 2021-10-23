import json
import re
from datetime import datetime, date, time
from enum import Enum
from contextlib import contextmanager
from dataclasses import fields as dc_fields, is_dataclass
from importlib import import_module
from typing import Optional, Any, Type, Callable, Union, ForwardRef, TypeVar,\
    Literal, List, Dict, Set, Tuple, cast, Generic

from .exceptions import UnknownTypeError
from .nodes import ArrayNode, BooleanNode, DictKeyType, DictNode,\
    LiteralNode, NullNode, NumberNode, ObjectNode, ReferenceNode,\
    RenderContext, StringNode, TupleNode, TypeNode, TypeVariableNode,\
    UnionNode, UnknownNode
from .utils import resolve_typevar


T = TypeVar('T')
BuilderT = TypeVar('BuilderT', bound='NodeBuilder')


class NodeBuilder:
    """NodeBuilder converts python objects to typescript's types."""

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


class NodeCompatibleClass(Generic[BuilderT]):
    @classmethod
    def convert_to_node(cls, builder: BuilderT) -> TypeNode:
        raise NotImplementedError()


class NodeCompatible(Generic[BuilderT]):
    def convert_to_node(self, builder: BuilderT) -> TypeNode:
        raise NotImplementedError()


__all__ = [
    'NodeBuilder',
    'NodeCompatibleClass',
    'NodeCompatible',
]
