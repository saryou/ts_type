# ts_type

pypi: https://pypi.org/project/ts-type/

## Overview

`ts_type` is a utility to generate typescript type definitions from python's objects.

This library uses python's dynamic nature to collect and construct type definitions. You can pick any objects which you wan to export. For example, assume that you have web api which statically typed with typing module and you can retrieve those types and urls. Use the information to generate union of api types easily like below:

Run this in REPL (python >= 3.9):


```python3
from dataclasses import dataclass
import ts_type as ts


@dataclass
class RequestForApiA:
    a: int
    b: str


@dataclass
class ResponseForApiA:
    result: bool


def api_a(request: RequestForApiA) -> ResponseForApiA:
    ...


@dataclass
class RequestForApiB:
    @dataclass
    class Item:
        id: int
        name: str

    items: list[Item]


def api_b(request: RequestForApiB) -> int:
    ...


apis = {
    '/api/a': api_a,
    '/api/b': api_b,
}


class Apis(ts.NodeCompatibleClass[ts.NodeBuilder]):
    @classmethod
    def convert_to_node(cls, builder: ts.NodeBuilder) -> ts.TypeNode:
        return ts.Union(of=[
            ts.Object(attrs=dict(
                # It must be `literal_to_node` because
                # `type_to_node` treats str as ForwardRef.
                url=builder.literal_to_node(url),

                # `type_to_node` supports many builtin types such as
                # dataclasses, Enums, Lists, Unions.
                request=builder.type_to_node(api.__annotations__['request']),
                response=builder.type_to_node(api.__annotations__['return']),
            ))
            for url, api in apis.items()
        ])


if __name__ == '__main__':
    ts.generator.add(Apis, 'apis', 'Apis')
    ts.generator.generate('output_dir')
```

output:

```typescript
export type Apis = {
    "url": "/api/a";
    "request": __main____RequestForApiA;
    "response": __main____ResponseForApiA;
} | {
    "url": "/api/b";
    "request": __main____RequestForApiB;
    "response": number;
};

type __main____RequestForApiA = {
    "a": number;
    "b": string;
};

type __main____ResponseForApiA = {
    "result": boolean;
};

type __main____RequestForApiB = {
    "items": __main____RequestForApiB__Item[];
};

type __main____RequestForApiB__Item = {
    "id": number;
    "name": string;
};
```

You can use it to extract request and response types from url.

```
type RequestForApiA = Extract<Apis, {url: "/api/a"}>['request'];
type ResponseForApiA = Extract<Apis, {url: "/api/a"}>['request'];
type RequestForApiB = Extract<Apis, {url: "/api/b"}>['request'];
type ResponseForApiB = Extract<Apis, {url: "/api/b"}>['request'];
```


## Basic Usage

1. add types to `TypeDefinitionGenerator`. (you can use `ts_type.generator`, which is an instance of the class)
2. call `TypeDefinitionGenerator.generate`

```py3
from dataclasses import dataclass
from typing import Optional, List, Dict

import ts_type as ts


@ts.gen_type
@dataclass
class MyCustomType:
    a: int
    b: Optional[str]
    c: List[str]
    d: Dict[str, int]


if __name__ == '__main__':
    ts.generator.generate('output_dir')
```


## Support Custom Types

You can customize builder to supports any objects. I use my own library [cleaned](https://github.com/saryou/cleaned) to validate requests. This is how I customize builder to support the library.


```python3
class Builder(ts.NodeBuilder):
    def handle_unknown_type(self, t: Any) -> ts.TypeNode:
        if isinstance(t, type):
            if issubclass(t, cl.Cleaned):
                return self.define_cleaned(t)
        return super().handle_unknown_type(t)

    def define_cleaned(self,
                       t: Type[cl.Cleaned],
                       exclude: set[str] = set()) -> ts.TypeNode:
        ret: ts.TypeNode = self.define_ref_node(t, lambda: ts.Object(
            attrs={
                k: self.type_to_node(self.field_to_type(f))
                for k, f in t._meta.fields.items()
            },
            omissible={k for k, f in t._meta.fields.items() if f.has_default}))

        if (exclude := exclude & set(t._meta.fields)):
            ret = ts.Omit(ret, ts.Union(
                [self.literal_to_node(k) for k in sorted(exclude)]))

        return ret

    def field_to_type(self, t: cl.Field) -> Type:
        if isinstance(t, cl.OptionalField):
            vt = self.field_to_type(t.field)
            return Union[None, cl.Undefined, vt]
        elif isinstance(t, cl.ListField):
            vt = self.field_to_type(t.value)
            return list[vt]
        elif isinstance(t, cl.SetField):
            vt = self.field_to_type(t.value)
            return set[vt]
        elif isinstance(t, cl.DictField):
            kt = self.field_to_type(t.key)
            vt = self.field_to_type(t.value)
            return dict[kt, vt]
        elif isinstance(t, cl.NestedField):
            return t._server()
        elif isinstance(t, cl.EnumField):
            return t._server()
        else:
            return ts.resolve_typevar(
                t.__class__,
                cl.Field.__parameters__[0])

# don't forget to pass the builder to generator
ts.generator.generate(output_dir, builder_cls=Builder)
```


## Other Examples

save following code as `example.py`

```py3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeVar, Generic, Optional

import ts_type as ts

T = TypeVar('T')


@ts.gen_type
@dataclass
class Article:
    class Type(Enum):
        news = 'news'
        essay = 'essay'

    @dataclass
    class Author:
        id: int
        name: str

    id: int
    type: Type
    title: str
    aurhot: Author
    published_at: datetime


@ts.gen_type
@dataclass
class Pagination(Generic[T]):
    items: list[T]
    next_cursor: Optional[str]


@ts.gen_type
@dataclass
class Response:
    status: int
    articles: Pagination[Article]
```


run:


```py3
import example
example.ts.generator.generate('output_dir')
```

output: `output_dir/example.gen.ts`

```typescript
export type Article = example__Article;

export type Pagination<T> = example__Pagination<T>;

export type Response = example__Response;

type example__Article = {
    "id": number;
    "type": example__Article__Type;
    "title": string;
    "aurhot": example__Article__Author;
    "published_at": string;
};

type example__Article__Type = "news" | "essay";

type example__Article__Author = {
    "id": number;
    "name": string;
};

type example__Pagination<T> = {
    "items": T[];
    "next_cursor": string | null;
};

type example__Response = {
    "status": number;
    "articles": example__Pagination<example__Article>;
};
```
