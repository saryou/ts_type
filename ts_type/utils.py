from types import GenericAlias
from typing import Optional, Any, TypeVar, List, Tuple


def resolve_typevar(cls: type, t: TypeVar) -> type:
    """resolve typevar to actual type or raise AssertionError

    It visits given type's mro and look into their __parameters__ to
    resolve actual type. I think this approach depends on internal
    implementations so it is fairly experimental.
    """
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
                        f'contains `{t}` itself (`{arg}` in `{m}`). '\
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


__all__ = [
    'resolve_typevar',
]
