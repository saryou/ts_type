from typing import Any, TypeVar, Iterable, List, Dict, Set, Sequence


class RenderContext:
    def __init__(self,
                 definitions: Dict[str, 'TypeNode'],
                 indent_level: int = 0,
                 indent_unit: str = ' ' * 4,):
        self.definitions = definitions
        self.indent_level = indent_level
        self.indent_unit = indent_unit

    def clone(self, **override) -> 'RenderContext':
        kwargs: Dict[str, Any] = dict(
            definitions=self.definitions,
            indent_level=self.indent_level,
            indent_unit=self.indent_unit)
        kwargs.update(override)
        return RenderContext(**kwargs)

    @property
    def indent(self) -> str:
        return self.indent_unit * self.indent_level

    def resolve_ref(self, node: 'TypeNode') -> 'TypeNode':
        if isinstance(node, ReferenceNode):
            return self.resolve_ref(self.definitions[node.identifier])
        return node


class TypeNode:
    def get_generic_params(self) -> List['TypeVariableNode']:
        return getattr(self, '__params__', [])

    def add_generic_params(self, variables: list['TypeVariableNode']):
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


class StringNode(BuiltinTypeNode, DictKeyType):
    def render(self, context: RenderContext) -> str:
        return 'string'

    def render_dict_key(self, context: RenderContext) -> str:
        return '[key: string]'


class NumberNode(BuiltinTypeNode, DictKeyType):
    def render(self, context: RenderContext) -> str:
        return 'number'

    def render_dict_key(self, context: RenderContext) -> str:
        return '[key: number]'


class BooleanNode(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'boolean'


class NullNode(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'null'


class UndefinedNode(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'undefined'


class UnknownNode(BuiltinTypeNode):
    def render(self, context: RenderContext) -> str:
        return 'unknown'


class LiteralNode(BuiltinTypeNode):
    def __init__(self, literal: str):
        self.literal = literal

    def render(self, context: RenderContext) -> str:
        return self.literal

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.literal == other.literal


class TypeVariableNode(BuiltinTypeNode):
    def __init__(self, typevar: TypeVar):
        self.typevar = typevar

    def render(self, context: RenderContext) -> str:
        return self.typevar.__name__

    def __eq__(self, other):
        return super().__eq__(other)\
            and self.typevar == other.typevar


class ReferenceNode(TypeNode):
    def __init__(self,
                 identifier: str,
                 typevars: List[TypeNode] = []):
        self.identifier = identifier
        self.typevars = typevars

    def render(self, context: RenderContext) -> str:
        ref_typevars = context.resolve_ref(self).get_generic_params()
        typevars = [*self.typevars, *ref_typevars[len(self.typevars):]]
        return self.identifier + self.__render_typevars(context, typevars)

    def __render_typevars(self,
                          context: RenderContext,
                          typevars: Sequence[TypeNode]) -> str:
        defs = [n.render(context) for n in typevars]
        return ''.join(['<', ', '.join(defs), '>']) if typevars else ''

    def __eq__(self, other):
        return isinstance(other, ReferenceNode)\
            and self.identifier == other.identifier\
            and self.typevars == other.typevars


class ObjectNode(TypeNode):
    def __init__(self,
                 attrs: Dict[str, TypeNode],
                 omissible: Set[str]):
        self.attrs = attrs
        self.omissible = omissible

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
        return isinstance(other, ObjectNode)\
            and self.attrs == other.attrs\
            and self.omissible == other.omissible


class TupleNode(TypeNode):
    def __init__(self, of: List[TypeNode]):
        self.of = of

    def render(self, context: RenderContext):
        return '[' + ', '.join([
            node.render(context) for node in self.of
        ]) + ']'

    def __eq__(self, other):
        return isinstance(other, TupleNode)\
            and self.of == other.of


class ArrayNode(TypeNode):
    def __init__(self, of: TypeNode):
        self.of = of

    def render(self, context: RenderContext):
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

    def render(self, context: RenderContext):
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

    def render(self, context: RenderContext):
        return '(' + ' | '.join([
            node.render(context) for node in self.of
        ]) + ')'

    def __eq__(self, other):
        return isinstance(other, UnionNode)\
            and self.of == other.of


__all__ = [
    'ArrayNode',
    'BooleanNode',
    'BuiltinTypeNode',
    'DictKeyType',
    'DictNode',
    'LiteralNode',
    'NullNode',
    'NumberNode',
    'ObjectNode',
    'ReferenceNode',
    'RenderContext',
    'StringNode',
    'TupleNode',
    'TypeNode',
    'TypeVariableNode',
    'UndefinedNode',
    'UnionNode',
    'UnknownNode',
]
