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
    Iterable, overload, Literal, List, Dict, Set, Tuple


T = TypeVar('T')


class Context:
    def __init__(self,
                 indent_level: int = 0,
                 indent_unit: str = ' ' * 4):
        self.indent_level = indent_level
        self.indent_unit = indent_unit

    def clone(self, **override) -> 'Context':
        kwargs: Dict[str, Any] = dict(
            indent_level=self.indent_level,
            indent_unit=self.indent_unit)
        kwargs.update(override)
        return Context(**kwargs)

    @property
    def indent(self) -> str:
        return self.indent_unit * self.indent_level


class TypeNode:
    def render(self, context: Context) -> str:
        raise NotImplementedError()


class DictKeyType(TypeNode):
    def render_dict_key(self, context: Context) -> str:
        raise NotImplementedError()


class BuiltinTypeNode(TypeNode):
    def __eq__(self, other):
        return type(self) is type(other)


class StringNode(BuiltinTypeNode, DictKeyType):
    def render(self, context: Context) -> str:
        return 'string'

    def render_dict_key(self, context: Context) -> str:
        return '[key: string]'


class NumberNode(BuiltinTypeNode, DictKeyType):
    def render(self, context: Context) -> str:
        return 'number'

    def render_dict_key(self, context: Context) -> str:
        return '[key: number]'


class BooleanNode(BuiltinTypeNode):
    def render(self, context: Context) -> str:
        return 'boolean'


class NullNode(BuiltinTypeNode):
    def render(self, context: Context) -> str:
        return 'null'


class UndefinedNode(BuiltinTypeNode):
    def render(self, context: Context) -> str:
        return 'undefined'


class UnknownNode(BuiltinTypeNode):
    def render(self, context: Context) -> str:
        return 'unknown'


class LiteralNode(BuiltinTypeNode):
    def __init__(self, literal: str):
        self.literal = literal

    def render(self, context: Context) -> str:
        return self.literal

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.literal == other.literal


class ReferenceNode(TypeNode):
    def __init__(self, identifier: str):
        self.identifier = identifier

    def render(self, context: Context) -> str:
        return self.identifier

    def __eq__(self, other):
        return isinstance(other, ReferenceNode)\
            and self.identifier == other.identifier


class ObjectNode(TypeNode):
    def __init__(self,
                 attrs: Dict[str, TypeNode],
                 omissible: Set[str]):
        self.attrs = attrs
        self.omissible = omissible

    def render(self, context: Context) -> str:
        c = context.clone(indent_level=context.indent_level + 1)
        return '\n'.join([
            '{',
            *[f'{c.indent}"{k}"{self._omissible_sign(k)}: {v.render(c)};'
              for k, v in self.attrs.items()],
            f'{context.indent}}}',
        ])

    def _omissible_sign(self, key: str) -> str:
        return '?' if key in self.omissible else ''

    def __eq__(self, other):
        return isinstance(other, ObjectNode)\
            and self.attrs == other.attrs\
            and self.omissible == other.omissible


class TupleNode(TypeNode):
    def __init__(self, of: List[TypeNode]):
        self.of = of

    def render(self, context: Context):
        return '[' + ', '.join([
            node.render(context) for node in self.of
        ]) + ']'

    def __eq__(self, other):
        return isinstance(other, TupleNode)\
            and self.of == other.of


class ArrayNode(TypeNode):
    def __init__(self, of: TypeNode):
        self.of = of

    def render(self, context: Context):
        return self.of.render(context) + '[]'

    def __eq__(self, other):
        return isinstance(other, ArrayNode)\
            and self.of == other.of


class DictNode(TypeNode):
    def __init__(self,
                 key: DictKeyType,
                 value: TypeNode):
        self.key = key
        self.value = value

    def render(self, context: Context):
        return ''.join([
            '{',
            self.key.render_dict_key(context),
            ': ',
            self.value.render(context),
            '}'
        ])

    def __eq__(self, other):
        return isinstance(other, DictNode)\
            and self.key == other.key\
            and self.value == other.value


class UnionNode(TypeNode):
    def __init__(self, of: List[TypeNode]):
        self.of = self._unique(self._flatten(of))

    def _flatten(self, of: List[TypeNode]) -> Iterable[TypeNode]:
        for node in of:
            if isinstance(node, UnionNode):
                yield from self._flatten(node.of)
            else:
                yield node

    def _unique(self, of: Iterable[TypeNode]) -> List[TypeNode]:
        ret: List[TypeNode] = []
        for node in of:
            if all(n != node for n in ret):
                ret.append(node)
        return ret

    def render(self, context: Context):
        return '(' + ' | '.join([
            node.render(context) for node in self.of
        ]) + ')'

    def __eq__(self, other):
        return isinstance(other, UnionNode)\
            and self.of == other.of


def _render_definitions(definitions: Dict[str, TypeNode],
                        ids_to_export: Set[str]) -> str:
    context = Context()

    def export(key: str) -> str:
        return 'export ' if key in ids_to_export else ''

    return '\n\n'.join([
        f'{export(k)}type {k} = {v.render(context)};'
        for k, v in definitions.items()
    ])


class NodeBuilder:
    def __init__(self,
                 default: Optional[Callable[
                     ['NodeBuilder', Any], TypeNode]] = None) -> None:
        self._modules: Dict[str, Any] = {}
        self._definitions: Dict[str, TypeNode] = {}
        self._stack: List[Tuple[str, str, Any]] = []
        self._default = default

    @property
    def definitions(self) -> Dict[str, Any]:
        return self._definitions

    def render_definitions(self, ids_to_export: Set[str] = set()) -> str:
        return _render_definitions(self._definitions, ids_to_export)

    def type_to_node(self, t: Any, allow_unknown: bool = False) -> TypeNode:
        try:
            return self._type_to_node(t, allow_unknown)
        except AssertionError:
            if allow_unknown:
                return UnknownNode()
            raise

    def _type_to_node(self, t: Any, unknown: bool) -> TypeNode:
        if t in [None, type(None)]:
            return NullNode()

        origin = getattr(t, '__origin__', None)
        args = getattr(t, '__args__', tuple())

        if origin is Union:
            assert args
            if len(args) == 1:
                return self.type_to_node(args[0], unknown)
            return UnionNode(
                of=[self.type_to_node(a, unknown) for a in args])
        elif origin is tuple:
            assert args
            return TupleNode([self.type_to_node(a, unknown) for a in args])
        elif origin is list or origin is set:
            assert args
            return ArrayNode(self.type_to_node(args[0], unknown))
        elif origin is dict:
            assert len(args) > 1
            key = self.type_to_node(args[0], unknown)
            assert isinstance(key, DictKeyType)
            return DictNode(
                key=key,
                value=self.type_to_node(args[1], unknown))
        elif origin is Literal:
            assert len(args) == 1
            return self._literal_to_node(args[0])
        elif isinstance(t, str):
            return self.type_to_node(self._resolve_annotation(t), unknown)
        elif isinstance(t, TypeVar):
            return self.type_to_node(self._resolve_typevar(t), unknown)
        elif isinstance(t, ForwardRef):
            _globals = self._current_module.__dict__
            evaluated = t._evaluate(_globals, None, set())  # type: ignore
            return self.type_to_node(evaluated, unknown)
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
                return self.define_ref_node(t, lambda: ObjectNode(
                    attrs={f.name: self.type_to_node(f.type, unknown)
                           for f in dc_fields(t)},
                    omissible=set()))

        return self.default(t)

    def define_ref_node(self,
                        type_or_id: Union[type, str],
                        define: Callable[[], TypeNode]) -> ReferenceNode:
        _id = '.'.join([type_or_id.__module__, type_or_id.__qualname__])\
            if isinstance(type_or_id, type) else\
            type_or_id

        _id = re.sub(r'\.', '__', _id)

        defs = self._definitions
        if _id not in defs:
            defs[_id] = TypeNode()

            try:
                if isinstance(type_or_id, type):
                    with self._begin_module_context(type_or_id):
                        defs[_id] = define()
                else:
                    defs[_id] = define()
            except Exception:
                del defs[_id]
                raise

        return ReferenceNode(_id)

    def _literal_to_node(self,
                         value: Union[int, bool, str, Enum]) -> LiteralNode:
        literal = value.name if isinstance(value, Enum) else value
        assert isinstance(literal, (int, bool, str))
        return LiteralNode(json.dumps(literal))

    def default(self, t: Any) -> TypeNode:
        if self._default is not None:
            ret = self._default(self, t)
            if isinstance(ret, TypeNode):
                return ret
            assert ret is None, 'default must return TypeNode or None'
        raise AssertionError(f'Type `{t}` is not supported.')

    @contextmanager
    def _begin_module_context(self, t: Type):
        self._import_module(t.__module__)
        self._stack.append((t.__module__, t.__qualname__, t))
        yield
        self._stack.pop()

    @property
    def _current_module(self) -> Any:
        return self._modules[self._stack[-1][0]]

    @property
    def _current(self) -> str:
        return '.'.join(self._stack[-1][:2])

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


class NodeCompatibleClass:
    @classmethod
    def convert_to_node(cls, builder: NodeBuilder) -> TypeNode:
        raise NotImplementedError()


class NodeCompatible:
    def convert_to_node(self, builder: NodeBuilder) -> TypeNode:
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
                 builder_cls: Type[NodeBuilder] = NodeBuilder):
        basedir = Path(output_dir)
        filepaths = {p for p in basedir.glob('**/*.gen.ts')}
        for _filepath, types in self.types.items():
            builder = builder_cls()
            defs: Dict[str, TypeNode] = dict()

            for name, value in types.items():
                if isinstance(value, TypeNode):
                    defs[name] = value
                else:
                    defs[name] = builder.type_to_node(value)

            filepath = basedir / (_filepath + '.gen.ts')
            filepaths.discard(filepath)

            _dirname = os.path.dirname(filepath)
            if not os.path.exists(_dirname):
                os.makedirs(_dirname)

            with open(filepath, 'w') as f:
                f.write(_render_definitions(defs, set(defs.keys())))
                f.write('\n\n\n')
                f.write(builder.render_definitions())

        for filepath in filepaths:
            os.remove(filepath)


generator = TypeDefinitionGenerator()


def gen_type(t: Type[T]) -> Type[T]:
    return generator.add(t)
