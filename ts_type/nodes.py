import typing


class RenderContext:
    def __init__(self,
                 definitions: typing.Dict[str, 'TypeNode'],
                 indent_level: int = 0,
                 indent_unit: str = ' ' * 4,):
        self.definitions = definitions
        self.indent_level = indent_level
        self.indent_unit = indent_unit

    def clone(self, **override) -> 'RenderContext':
        kwargs: typing.Dict[str, typing.Any] = dict(
            definitions=self.definitions,
            indent_level=self.indent_level,
            indent_unit=self.indent_unit)
        kwargs.update(override)
        return RenderContext(**kwargs)

    @property
    def indent(self) -> str:
        return self.indent_unit * self.indent_level

    def resolve_ref(self, node: 'TypeNode') -> 'TypeNode':
        if isinstance(node, Reference):
            return self.resolve_ref(self.definitions[node.identifier])
        return node


class TypeNode:
    def get_generic_params(self) -> typing.List['TypeVariable']:
        return getattr(self, '__params__', [])

    def add_generic_params(self, variables: typing.List['TypeVariable']):
        self.__params__ = self.get_generic_params()
        self.__params__.extend(variables)

    def render(self, context: RenderContext) -> str:
        raise NotImplementedError()


class DictKeyType(TypeNode):
    def render_dict_key(self, context: RenderContext) -> str:
        raise NotImplementedError()


class BuiltinTypeNode(TypeNode):
    def __eq__(self, other):
        return type(self) is type(other)


class String(BuiltinTypeNode, DictKeyType):
    def render(self, context: RenderContext) -> str:
        return 'string'

    def render_dict_key(self, context: RenderContext) -> str:
        return '[key: string]'


class Number(BuiltinTypeNode, DictKeyType):
    def render(self, context: RenderContext) -> str:
        return 'number'

    def render_dict_key(self, context: RenderContext) -> str:
        return '[key: number]'


class Boolean(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'boolean'


class Null(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'null'


class Undefined(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'undefined'


class Unknown(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'unknown'


class Literal(BuiltinTypeNode):
    def __init__(self, literal: str):
        self.literal = literal

    def render(self, context: RenderContext) -> str:
        return self.literal

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.literal == other.literal


class TypeVariable(BuiltinTypeNode):
    def __init__(self, typevar: typing.TypeVar):
        self.typevar = typevar

    def render(self, context: RenderContext) -> str:
        return self.typevar.__name__

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.typevar == other.typevar


class Reference(TypeNode):
    def __init__(self,
                 identifier: str,
                 typevars: typing.List[TypeNode] = []):
        self.identifier = identifier
        self.typevars = typevars

    def render(self, context: RenderContext) -> str:
        ref_typevars = context.resolve_ref(self).get_generic_params()
        typevars = [*self.typevars, *ref_typevars[len(self.typevars):]]
        return self.identifier + self.__render_typevars(context, typevars)

    def __render_typevars(self,
                          context: RenderContext,
                          typevars: typing.Sequence[TypeNode]) -> str:
        defs = [n.render(context) for n in typevars]
        return ''.join(['<', ', '.join(defs), '>']) if typevars else ''

    def __eq__(self, other):
        return isinstance(other, Reference)\
            and self.identifier == other.identifier\
            and self.typevars == other.typevars


class Object(TypeNode):
    def __init__(self,
                 attrs: typing.Dict[str, TypeNode],
                 omissible: typing.Optional[typing.Set[str]] = None):
        self.attrs = attrs
        self.omissible = omissible or set()

    def render(self, context: RenderContext) -> str:
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
        return isinstance(other, Object)\
            and self.attrs == other.attrs\
            and self.omissible == other.omissible


class Tuple(TypeNode):
    def __init__(self, of: typing.List[TypeNode]):
        self.of = of

    def render(self, context: RenderContext):
        return '[' + ', '.join([
            node.render(context) for node in self.of
        ]) + ']'

    def __eq__(self, other):
        return isinstance(other, Tuple)\
            and self.of == other.of


class Array(TypeNode):
    def __init__(self, of: TypeNode):
        self.of = of

    def render(self, context: RenderContext):
        return _render_with_parenthesis(self.of, context) + '[]'

    def __eq__(self, other):
        return isinstance(other, Array)\
            and self.of == other.of


class Dict(TypeNode):
    def __init__(self,
                 key: DictKeyType,
                 value: TypeNode):
        self.key = key
        self.value = value

    def render(self, context: RenderContext):
        return ''.join([
            '{',
            self.key.render_dict_key(context),
            ': ',
            self.value.render(context),
            '}'
        ])

    def __eq__(self, other):
        return isinstance(other, Dict)\
            and self.key == other.key\
            and self.value == other.value


class Union(TypeNode):
    def __init__(self, of: typing.List[TypeNode]):
        self.of = self._unique(self._flatten(of))
        assert self.of

    def _flatten(self, of: typing.List[TypeNode]) -> typing.Iterable[TypeNode]:
        for node in of:
            if isinstance(node, Union):
                yield from self._flatten(node.of)
            else:
                yield node

    def _unique(self, of: typing.Iterable[TypeNode]) -> typing.List[TypeNode]:
        ret: typing.List[TypeNode] = []
        for node in of:
            if all(n != node for n in ret):
                ret.append(node)
        return ret

    def render(self, context: RenderContext):
        return ' | '.join([
            _render_with_parenthesis(node, context) for node in self.of
        ])

    def __eq__(self, other):
        return isinstance(other, Union)\
            and self.of == other.of


class Intersection(TypeNode):
    def __init__(self, of: typing.List[TypeNode]):
        self.of = self._unique(self._flatten(of))
        assert self.of

    def _flatten(self, of: typing.List[TypeNode]) -> typing.Iterable[TypeNode]:
        for node in of:
            if isinstance(node, Intersection):
                yield from self._flatten(node.of)
            else:
                yield node

    def _unique(self, of: typing.Iterable[TypeNode]) -> typing.List[TypeNode]:
        ret: typing.List[TypeNode] = []
        for node in of:
            if all(n != node for n in ret):
                ret.append(node)
        return ret

    def render(self, context: RenderContext):
        return ' | '.join([
            _render_with_parenthesis(node, context) for node in self.of
        ])

    def __eq__(self, other):
        return isinstance(other, Intersection)\
            and self.of == other.of


class Keyof(TypeNode):
    def __init__(self, of: TypeNode):
        self.of = of

    def render(self, context: RenderContext):
        return 'keyof ' + _render_with_parenthesis(self.of, context)

    def __eq__(self, other):
        return isinstance(other, Tuple)\
            and self.of == other.of


class CustomNode(TypeNode):
    def __init__(self, name: str, parameters: typing.List[TypeNode]):
        self.name = name
        self.parameters = parameters

    def render(self, context: RenderContext):
        return f'{self.name}<' + ', '.join([
            node.render(context) for node in self.parameters
        ]) + '>'

    def __eq__(self, other):
        return isinstance(other, self.__class__)\
            and self.name == other.name\
            and self.parameters == other.parameters


class Partial(CustomNode):
    def __init__(self, type: TypeNode):
        super().__init__('Partial', [type])


class Required(CustomNode):
    def __init__(self, type: TypeNode):
        super().__init__('Required', [type])


class Readonly(CustomNode):
    def __init__(self, type: TypeNode):
        super().__init__('Readonly', [type])


class Record(CustomNode):
    def __init__(self, keys: TypeNode, type: TypeNode):
        super().__init__('Record', [keys, type])


class Pick(CustomNode):
    def __init__(self, type: TypeNode, keys: TypeNode):
        super().__init__('Pick', [type, keys])


class Omit(CustomNode):
    def __init__(self, type: TypeNode, keys: TypeNode):
        super().__init__('Omit', [type, keys])


class Exclude(CustomNode):
    def __init__(self, type: TypeNode, excluded_union: TypeNode):
        super().__init__('Exclude', [type, excluded_union])


class Extract(CustomNode):
    def __init__(self, type: TypeNode, union: TypeNode):
        super().__init__('Extract', [type, union])


class NonNullable(CustomNode):
    def __init__(self, type: TypeNode):
        super().__init__('NonNullable', [type])


def _render_with_parenthesis(node: TypeNode, context: RenderContext) -> str:
    expr = node.render(context)

    if isinstance(node, (Union, Intersection)):
        if len(node.of) == 1:
            return expr
        return f'({expr})'

    if isinstance(node, Keyof):
        return f'({expr})'

    return expr


__all__ = [
    'Array',
    'Boolean',
    'BuiltinTypeNode',
    'CustomNode',
    'Dict',
    'DictKeyType',
    'Exclude',
    'Extract',
    'Intersection',
    'Keyof',
    'Literal',
    'NonNullable',
    'Null',
    'Number',
    'Object',
    'Omit',
    'Partial',
    'Pick',
    'Readonly',
    'Record',
    'Reference',
    'RenderContext',
    'Required',
    'String',
    'Tuple',
    'TypeNode',
    'TypeVariable',
    'Undefined',
    'Union',
    'Unknown',
]
