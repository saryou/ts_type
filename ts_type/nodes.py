import typing
import contextlib


class RenderContext:
    def __init__(self,
                 definitions: typing.Dict[str, 'TypeNode'],
                 indent_level: int = 0,
                 indent_unit: str = ' ' * 4,):
        self.definitions = definitions
        self.indent_level = indent_level
        self.indent_unit = indent_unit
        self.typevar_definition_enabled = False

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
            return self.resolve_ref(
                self.definitions.get(node.identifier, Never()))
        return node

    def resolve_typevars(self, node: 'TypeNode')\
            -> typing.List['TypeVariable']:
        if isinstance(node, Reference):
            node = self.definitions.get(node.identifier, Never())
        return node.get_generic_params()

    def render_typevars(
            self,
            typevars: typing.Union[
                'TypeNode',
                typing.Sequence[typing.Union['TypeVariable', 'TypeNode']],
            ],
            brackets: bool = True) -> str:
        if isinstance(typevars, TypeNode):
            typevars = self.resolve_typevars(typevars)
        defs = ', '.join([n.render(self) for n in typevars])
        if not brackets:
            return defs
        return ''.join(['<', defs, '>']) if typevars else ''

    @contextlib.contextmanager
    def enable_typevar_definition(self):
        self.typevar_definition_enabled = True
        yield
        self.typevar_definition_enabled = False


_empty_context = RenderContext({})
_empty_context.typevar_definition_enabled = True


class TypeNode:
    def get_generic_params(self) -> typing.List['TypeVariable']:
        proxy = self.get_proxy_for_generic_params()
        if proxy is not self:
            return proxy.get_generic_params()

        return getattr(self, '__params__', [])

    def add_generic_params(self, variables: typing.List['TypeVariable']):
        proxy = self.get_proxy_for_generic_params()
        if proxy is not self:
            proxy.add_generic_params(variables)
            return

        self.__params__ = self.get_generic_params()
        self.__params__.extend(variables)

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self

    def render(self, context: RenderContext) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.render(_empty_context)

    def __repr__(self) -> str:
        return f'<{self.__class__}: {self}>'


class DictKeyType(TypeNode):
    def render_dict_key(self, context: RenderContext) -> str:
        return f'[K in {self.render(context)}]'


class GlobalTypeNode(TypeNode):
    def __eq__(self, other):
        return type(self) is type(other)


class String(GlobalTypeNode, DictKeyType):
    def render(self, context: RenderContext) -> str:
        return 'string'

    def render_dict_key(self, context: RenderContext) -> str:
        return '[key: string]'


class Number(GlobalTypeNode, DictKeyType):
    def render(self, context: RenderContext) -> str:
        return 'number'

    def render_dict_key(self, context: RenderContext) -> str:
        return '[key: number]'


class Boolean(GlobalTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'boolean'


class Null(GlobalTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'null'


class Undefined(GlobalTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'undefined'


class Unknown(GlobalTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'unknown'


class Never(GlobalTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'never'


class Literal(GlobalTypeNode):
    def __init__(self, literal: str):
        self.literal = literal

    def render(self, context: RenderContext) -> str:
        return self.literal

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.literal == other.literal


class TypeVariable(GlobalTypeNode):
    def __init__(self,
                 typevar: typing.TypeVar,
                 condition: typing.Optional[TypeNode] = None,
                 default: typing.Optional[TypeNode] = None):
        self.typevar = typevar
        self.condition = condition
        self.default = default

    def render(self, context: RenderContext) -> str:
        ret = self.typevar.__name__
        if context.typevar_definition_enabled:
            if self.condition is not None:
                ret = f'{ret} extends {self.condition.render(context)}'
            if self.default is not None:
                ret = f'{ret} = {self.default.render(context)}'
        return ret

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.typevar == other.typevar


class Reference(DictKeyType):
    def __init__(self,
                 identifier: str,
                 typevars: typing.List[TypeNode] = []):
        self.identifier = identifier
        self.typevars = typevars

    def render(self, context: RenderContext) -> str:
        ref_typevars = context.resolve_typevars(self)
        typevars = [*self.typevars, *ref_typevars[len(self.typevars):]]
        return self.identifier + context.render_typevars(typevars)

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

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.of.get_proxy_for_generic_params()


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

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.value.get_proxy_for_generic_params()


class Union(DictKeyType):
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
        return ' & '.join([
            _render_with_parenthesis(node, context) for node in self.of
        ])

    def __eq__(self, other):
        return isinstance(other, Intersection)\
            and self.of == other.of


class Keyof(DictKeyType):
    def __init__(self, of: TypeNode):
        self.of = of

    def render(self, context: RenderContext):
        return 'keyof ' + _render_with_parenthesis(self.of, context)

    def __eq__(self, other):
        return isinstance(other, Tuple)\
            and self.of == other.of

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.of.get_proxy_for_generic_params()


class Lookup(DictKeyType):
    def __init__(self, node: TypeNode, lookup: TypeNode):
        self.node = node
        self.lookup = lookup

    def render(self, context: RenderContext) -> str:
        return _render_with_parenthesis(self.node, context)\
            + f'[{self.lookup.render(context)}]'

    def __eq__(self, other):
        return isinstance(other, Lookup)\
            and self.node == other.node\
            and self.lookup == other.lookup

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.node.get_proxy_for_generic_params()


class UtilityNode(TypeNode):
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


class Partial(UtilityNode):
    def __init__(self, type: TypeNode):
        self.type = type
        super().__init__('Partial', [type])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Required(UtilityNode):
    def __init__(self, type: TypeNode):
        self.type = type
        super().__init__('Required', [type])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Readonly(UtilityNode):
    def __init__(self, type: TypeNode):
        self.type = type
        super().__init__('Readonly', [type])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Record(UtilityNode):
    def __init__(self, keys: TypeNode, type: TypeNode):
        super().__init__('Record', [keys, type])


class Pick(UtilityNode):
    def __init__(self, type: TypeNode, keys: TypeNode):
        self.type = type
        super().__init__('Pick', [type, keys])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Omit(UtilityNode):
    def __init__(self, type: TypeNode, keys: TypeNode):
        self.type = type
        super().__init__('Omit', [type, keys])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Exclude(UtilityNode):
    def __init__(self, type: TypeNode, excluded_union: TypeNode):
        self.type = type
        super().__init__('Exclude', [type, excluded_union])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Extract(UtilityNode):
    def __init__(self, type: TypeNode, union: TypeNode):
        self.type = type
        super().__init__('Extract', [type, union])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class NonNullable(UtilityNode):
    def __init__(self, type: TypeNode):
        self.type = type
        super().__init__('NonNullable', [type])

    def get_proxy_for_generic_params(self) -> 'TypeNode':
        return self.type.get_proxy_for_generic_params()


class Infer(TypeNode):
    def __init__(self, typevar: typing.TypeVar):
        self.typevar = typevar

    def render(self, context: RenderContext) -> str:
        return f'infer {self.typevar.__name__}'


class Conditional(TypeNode):
    def __init__(self,
                 type: TypeNode,
                 condition: TypeNode,
                 true: TypeNode,
                 false: TypeNode):
        self.type = type
        self.condition = condition
        self.true = true
        self.false = false

    def render(self, context: RenderContext):
        type = self.type.render(context)
        condition = self.condition.render(context)
        true = self.true.render(context)
        false = self.false.render(context)
        return f'{type} extends {condition} ? {true} : {false}'

    def __eq__(self, other):
        return isinstance(other, self.__class__)\
            and self.type == other.type\
            and self.condition == other.condition\
            and self.true == other.true\
            and self.false == other.false


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
    'Conditional',
    'Dict',
    'DictKeyType',
    'Exclude',
    'Extract',
    'GlobalTypeNode',
    'Infer',
    'Intersection',
    'Keyof',
    'Literal',
    'Lookup',
    'Never',
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
    'UtilityNode',
]
