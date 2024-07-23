from dataclasses import dataclass
from datetime import datetime, date, time
from enum import Enum
from typing import Union, Literal, List, Set, Tuple, Dict, Optional, TypeVar, \
    Generic
from unittest import TestCase

import ts_type as ts


T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
W = TypeVar('W')


class Tests(TestCase):
    def test_primitives(self):
        builder = ts.NodeBuilder()

        self.assertIsInstance(builder.type_to_node(str), ts.String)
        self.assertIsInstance(builder.type_to_node(datetime), ts.String)
        self.assertIsInstance(builder.type_to_node(date), ts.String)
        self.assertIsInstance(builder.type_to_node(time), ts.String)
        self.assertIsInstance(builder.type_to_node(int), ts.Number)
        self.assertIsInstance(builder.type_to_node(float), ts.Number)

        try:
            from typing import Never  # type: ignore
            self.assertIsInstance(builder.type_to_node(Never), ts.Never)
        except ImportError:
            try:
                from typing_extensions import Never  # type: ignore
                self.assertIsInstance(builder.type_to_node(Never), ts.Never)
            except Exception:
                pass

    def test_literals(self):
        builder = ts.NodeBuilder()

        a = builder.type_to_node(Literal['a'])
        assert isinstance(a, ts.Literal)
        self.assertEqual(a.literal, '"a"')

        true = builder.type_to_node(Literal[True])
        assert isinstance(true, ts.Literal)
        self.assertEqual(true.literal, 'true')

        one = builder.type_to_node(Literal[1])
        assert isinstance(one, ts.Literal)
        self.assertEqual(one.literal, '1')

        a_or_true_or_one = builder.type_to_node(Literal['a', True, 1])
        assert isinstance(a_or_true_or_one, ts.Union)
        assert isinstance(a_or_true_or_one.of[0], ts.Literal)
        self.assertEqual(a_or_true_or_one.of[0].literal, '"a"')
        assert isinstance(a_or_true_or_one.of[1], ts.Literal)
        self.assertEqual(a_or_true_or_one.of[1].literal, 'true')
        assert isinstance(a_or_true_or_one.of[2], ts.Literal)
        self.assertEqual(a_or_true_or_one.of[2].literal, '1')

    def test_tuple(self):
        builder = ts.NodeBuilder()

        tup = builder.type_to_node(Tuple[str, int])
        assert isinstance(tup, ts.Tuple)
        assert isinstance(tup.of[0], ts.String)
        assert isinstance(tup.of[1], ts.Number)

    def test_union(self):
        builder = ts.NodeBuilder()

        union1 = builder.type_to_node(Union[str, Literal[1]])
        assert isinstance(union1, ts.Union)
        assert isinstance(union1.of[0], ts.String)
        assert isinstance(union1.of[1], ts.Literal)
        self.assertEqual(union1.of[1].literal, '1')

        try:
            union_type = eval('int | str')
        except TypeError:
            pass
        else:
            union2 = builder.type_to_node(union_type)
            assert isinstance(union2, ts.Union)
            assert isinstance(union2.of[0], ts.Number)
            assert isinstance(union2.of[1], ts.String)

    def test_array(self):
        builder = ts.NodeBuilder()

        _list = builder.type_to_node(List[str])
        assert isinstance(_list, ts.Array)
        assert isinstance(_list.of, ts.String)

        _set = builder.type_to_node(Set[float])
        assert isinstance(_set, ts.Array)
        assert isinstance(_set.of, ts.Number)

    def test_dict(self):
        builder = ts.NodeBuilder()

        # keys
        str_key = builder.type_to_node(Dict[str, bool])
        assert isinstance(str_key, ts.Dict)
        assert isinstance(str_key.key, ts.String)
        assert isinstance(str_key.value, ts.Boolean)

        int_key = builder.type_to_node(Dict[int, int])
        assert isinstance(int_key, ts.Dict)
        assert isinstance(int_key.key, ts.Number)
        assert isinstance(int_key.value, ts.Number)

        float_key = builder.type_to_node(Dict[float, List[int]])
        assert isinstance(float_key, ts.Dict)
        assert isinstance(float_key.key, ts.Number)
        assert isinstance(float_key.value, ts.Array)
        assert isinstance(float_key.value.of, ts.Number)

        # key type must be string or number
        with self.assertRaises(AssertionError):
            builder.type_to_node(Dict[bool, str])

    def test_enum(self):
        builder = ts.NodeBuilder()

        class EnumA(Enum):
            a = 1
            b = '2'

        a_ref = builder.type_to_node(EnumA)
        assert isinstance(a_ref, ts.Reference)
        a = builder.definitions[a_ref.identifier]
        assert isinstance(a, ts.Union)
        assert isinstance(a.of[0], ts.Literal)
        self.assertEqual(a.of[0].literal, '"a"')
        assert isinstance(a.of[1], ts.Literal)
        self.assertEqual(a.of[1].literal, '"b"')

    def test_dataclass(self):
        builder = ts.NodeBuilder()

        @dataclass
        class A:
            a: int
            b: str
            c: List[Set[int]]

        a_ref = builder.type_to_node(A)
        assert isinstance(a_ref, ts.Reference)
        a = builder.definitions[a_ref.identifier]
        assert isinstance(a, ts.Object)
        assert isinstance(a.attrs['a'], ts.Number)
        assert isinstance(a.attrs['b'], ts.String)
        assert isinstance(a.attrs['c'], ts.Array)
        assert isinstance(a.attrs['c'].of, ts.Array)
        assert isinstance(a.attrs['c'].of.of, ts.Number)

    def test_complicated(self):
        builder = ts.NodeBuilder()

        rec_a_ref = builder.type_to_node(RecursiveA)
        assert isinstance(rec_a_ref, ts.Reference)

        rec_b_ref = builder.type_to_node(RecursiveB)
        assert isinstance(rec_b_ref, ts.Reference)

        # RecursiveA
        rec_a = builder.definitions[rec_a_ref.identifier]
        assert isinstance(rec_a, ts.Object)

        assert isinstance(rec_a.attrs['children1'], ts.Array)
        assert isinstance(rec_a.attrs['children1'].of, ts.Reference)
        self.assertEqual(rec_a.attrs['children1'].of.identifier,
                         rec_a_ref.identifier)

        assert isinstance(rec_a.attrs['children2'], ts.Dict)
        assert isinstance(rec_a.attrs['children2'].key, ts.String)
        assert isinstance(rec_a.attrs['children2'].value, ts.Reference)
        self.assertEqual(rec_a.attrs['children2'].value.identifier,
                         rec_a_ref.identifier)

        assert isinstance(rec_a.attrs['children3'], ts.Union)
        assert isinstance(rec_a.attrs['children3'].of[0], ts.Array)
        assert isinstance(rec_a.attrs['children3'].of[0].of, ts.Reference)
        self.assertEqual(rec_a.attrs['children3'].of[0].of.identifier,
                         rec_a_ref.identifier)
        assert isinstance(rec_a.attrs['children3'].of[1], ts.Reference)
        self.assertEqual(rec_a.attrs['children3'].of[1].identifier,
                         rec_b_ref.identifier)

        # RecursiveB
        rec_b = builder.definitions[rec_b_ref.identifier]
        assert isinstance(rec_b, ts.Object)

        assert isinstance(rec_b.attrs['recursive'], ts.Reference)
        self.assertEqual(rec_b.attrs['recursive'].identifier,
                         rec_a_ref.identifier)

        assert isinstance(rec_b.attrs['tuples'], ts.Dict)
        assert isinstance(rec_b.attrs['tuples'].key, ts.String)
        assert isinstance(rec_b.attrs['tuples'].value, ts.Array)
        assert isinstance(rec_b.attrs['tuples'].value.of, ts.Tuple)
        assert isinstance(rec_b.attrs['tuples'].value.of.of[0], ts.Reference)
        self.assertEqual(rec_b.attrs['tuples'].value.of.of[0].identifier,
                         rec_a_ref.identifier)
        assert isinstance(rec_b.attrs['tuples'].value.of.of[1], ts.Reference)
        self.assertEqual(rec_b.attrs['tuples'].value.of.of[1].identifier,
                         rec_b_ref.identifier)
        assert isinstance(rec_b.attrs['tuples'].value.of.of[2], ts.Number)

    def test_generics(self):
        builder = ts.NodeBuilder()

        rec_a_ref = builder.type_to_node(RecursiveA)
        assert isinstance(rec_a_ref, ts.Reference)

        rec_b_ref = builder.type_to_node(RecursiveB)
        assert isinstance(rec_b_ref, ts.Reference)

        # GenericA
        a_ref = builder.type_to_node(GenericA)
        assert isinstance(a_ref, ts.Reference)
        a = builder.definitions[a_ref.identifier]
        assert isinstance(a, ts.Object)

        assert isinstance(a.attrs['t'], ts.Reference)
        self.assertEqual(a.attrs['t'].identifier, rec_a_ref.identifier)

        # GenericB
        b_ref = builder.type_to_node(GenericB)
        assert isinstance(b_ref, ts.Reference)
        b = builder.definitions[b_ref.identifier]
        assert isinstance(b, ts.Object)

        assert isinstance(b.attrs['t'], ts.Reference)
        self.assertEqual(b.attrs['t'].identifier, rec_b_ref.identifier)

        # GenericE
        e_ref = builder.type_to_node(GenericE)
        assert isinstance(e_ref, ts.Reference)
        e = builder.definitions[e_ref.identifier]
        assert isinstance(e, ts.Object)

        # GenericC makes T as Tuple[dict[str, U], V, W]
        assert isinstance(e.attrs['t'], ts.Tuple)
        self.assertEqual(len(e.attrs['t'].of), 3)
        assert isinstance(e.attrs['t'].of[0], ts.Dict)
        assert isinstance(e.attrs['t'].of[0].key, ts.String)

        # GenericD makes U as int, V as str
        assert isinstance(e.attrs['t'].of[0].value, ts.Number)
        assert isinstance(e.attrs['t'].of[1], ts.String)
        self.assertEqual(e.attrs['u'], e.attrs['t'].of[0].value)
        self.assertEqual(e.attrs['v'], e.attrs['t'].of[1])

        # GenericE makes W as Optional[str]
        assert isinstance(e.attrs['t'].of[2], ts.Union)
        assert any(isinstance(t, ts.Null) for t in e.attrs['t'].of[2].of)
        assert any(isinstance(t, ts.String) for t in e.attrs['t'].of[2].of)
        self.assertEqual(e.attrs['w'], e.attrs['t'].of[2])
        assert isinstance(e.attrs['w'], ts.Union)

        assert isinstance(e.attrs['any'], ts.Union)
        self.assertTrue(any(t == e.attrs['u'] for t in e.attrs['any'].of))
        self.assertTrue(any(t == e.attrs['v'] for t in e.attrs['any'].of))
        self.assertTrue(all(any(t == w for t in e.attrs['any'].of)
                            for w in e.attrs['w'].of))

    def test_parametarized_generics(self):
        builder = ts.NodeBuilder()

        # single parameter
        a_ref = builder.type_to_node(GenericTest[int])
        assert isinstance(a_ref, ts.Reference)
        self.assertEqual(a_ref.typevars, [ts.Number()])

        a = builder.definitions[a_ref.identifier]
        assert isinstance(a, ts.Object)
        self.assertEqual(len(a.attrs), 1)
        self.assertEqual(a.attrs['t'], ts.TypeVariable(T))
        self.assertEqual(a.get_generic_params(), [ts.TypeVariable(T)])

        # multiple parameters
        c_ref = builder.type_to_node(GenericC[int, str, bool])
        assert isinstance(c_ref, ts.Reference)
        self.assertEqual(c_ref.typevars, [
            ts.Number(),
            ts.String(),
            ts.Boolean(),
        ])

        c = builder.definitions[c_ref.identifier]
        assert isinstance(c, ts.Object)
        self.assertEqual(len(c.attrs), 5)
        self.assertEqual(c.attrs['t'], ts.Tuple([
            ts.Dict(ts.String(), ts.TypeVariable(U)),
            ts.TypeVariable(V),
            ts.TypeVariable(W),
        ]))
        self.assertEqual(c.attrs['u'], ts.TypeVariable(U))
        self.assertEqual(c.attrs['v'], ts.TypeVariable(V))
        self.assertEqual(c.attrs['w'], ts.TypeVariable(W))
        self.assertEqual(c.attrs['any'], ts.Union([
            ts.TypeVariable(U),
            ts.TypeVariable(V),
            ts.TypeVariable(W),
        ]))
        self.assertEqual(c.get_generic_params(), [
            ts.TypeVariable(U),
            ts.TypeVariable(V),
            ts.TypeVariable(W),
        ])

        d_ref = builder.type_to_node(GenericD[List[int]])
        assert isinstance(d_ref, ts.Reference)
        self.assertEqual(d_ref.typevars, [
            ts.Array(ts.Number()),
        ])

        d = builder.definitions[d_ref.identifier]
        assert isinstance(d, ts.Object)
        self.assertEqual(len(d.attrs), 5)
        self.assertEqual(d.attrs['t'], ts.Tuple([
            ts.Dict(ts.String(), ts.Number()),
            ts.String(),
            ts.TypeVariable(W),
        ]))
        self.assertEqual(d.attrs['u'], ts.Number())
        self.assertEqual(d.attrs['v'], ts.String())
        self.assertEqual(d.attrs['w'], ts.TypeVariable(W))
        self.assertEqual(d.attrs['any'], ts.Union([
            ts.Number(),
            ts.String(),
            ts.TypeVariable(W),
        ]))
        self.assertEqual(d.get_generic_params(), [
            ts.TypeVariable(W),
        ])


@dataclass
class RecursiveA:
    children1: List['RecursiveA']
    children2: Dict[str, 'RecursiveA']
    children3: Union[List['RecursiveA'], 'RecursiveB']


@dataclass
class RecursiveB:
    recursive: RecursiveA
    tuples: Dict[str, List[Tuple[RecursiveA, 'RecursiveB', int]]]


@dataclass
class GenericTest(Generic[T]):
    t: T


@dataclass
class GenericA(GenericTest[RecursiveA]):
    pass


@dataclass
class GenericB(GenericTest[RecursiveB]):
    pass


@dataclass
class GenericC(GenericTest[Tuple[Dict[str, U], V, W]]):
    u: U
    v: V
    w: W
    any: Union[U, V, W]


@dataclass
class GenericD(GenericC[int, str, W]):
    pass


@dataclass
class GenericE(GenericD[Optional[str]]):
    pass
