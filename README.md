# ts_type

## Overview

`ts_type` is a Python library that generates TypeScript type definitions from Python type hints. This library helps bridge the gap between backend (Python) and frontend (TypeScript), ensuring type safety and improving development efficiency.

- **GitHub**: [https://github.com/saryou/ts_type](https://github.com/saryou/ts_type)  
- **PyPI**: [https://pypi.org/project/ts-type/](https://pypi.org/project/ts-type/)

## Motivation

In modern Python development, it has become common to use `typing` to annotate type information. On the frontend side, TypeScript dominates the ecosystem due to its strong typing capabilities. Many projects use Python as the backend and TypeScript as the frontend, making it highly beneficial to share type information between the two.

At **[Nailbook](https://nailbook.jp/)**, we use Python for the backend and TypeScript for the frontend, making `ts_type` a perfect fit for our needs. By generating TypeScript types from all API definitions, we not only prevent type mismatches in API calls but also achieve **exceptionally high development efficiency**.

## What is ts_type?

`ts_type` is a utility for generating TypeScript type definitions from Python objects.

This library leverages Python's dynamic nature to collect and construct type definitions.  
You can specify any objects you want to export as TypeScript types.  

For example, if you have a web API that uses Python’s `typing` module for static type annotations,  
you can retrieve those type definitions and generate a union of API types effortlessly:

## Why Use `ts_type`?

- **Automatic Type Synchronization**  
  Ensure that your frontend TypeScript types always match your backend Python types.
- **Eliminate Type Mismatches**  
  Reduce runtime errors caused by incorrect API usage.
- **Improve Developer Productivity**  
  No need to manually write or update TypeScript types for API responses.

## How To Use

### Installation

You can install `ts_type` via pip:

```sh
pip install ts-type
```


### Generating TypeScript Definitions from Python Models

To use `ts_type`, follow these steps:

#### 1. Define Python Models

Create a file named `example.py` and define your Python models:

```python
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
```

#### 2. Create a code generation script

Next, create a script named `codegen.py` to generate TypeScript types:

```py3
# Ensure that the defined class is registered
import example

import ts_type as ts
ts.generator.generate('output_dir')
```

#### 3. Run the script

Execute the script to generate TypeScript types:

```sh
python codegen.py
```

This will generate `output_dir/example.gen.ts` with the following content:

```typescript
export type MyCustomType = example__MyCustomType;

type example__MyCustomType = {
    "a": number;
    "b": string | null;
    "c": string[];
    "d": {[key: string]: number};
};
```


### Support Custom Types

You can customize the builder to support any objects, including third-party libraries. For example, if you are using the [`cleaned`](https://github.com/saryou/cleaned) library (**which, let's be honest, might only be me using it 😇**) for request validation, you can define a custom builder as follows:

```python3
class Builder(ts.NodeBuilder):
    def handle_unknown_type(self, t: Any) -> ts.TypeNode:
        if isinstance(t, type):
            if issubclass(t, cl.Cleaned):
                return self.define_cleaned(t)
            if issubclass(t, cl.Undefined):
                return ts.Undefined()
        return super().handle_unknown_type(t)

    def define_cleaned(self,
                       t: type[cl.Cleaned],
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

    def field_to_type(self, t: cl.Field) -> type:
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

# Don't forget to pass the builder to the generator
ts.generator.generate(output_dir, builder_cls=Builder)
```

### Generating TypeScript Definitions from API Models

Below is an example demonstrating how `ts_type` can generate TypeScript type definitions for API request/response models:

#### 1. Define API models
Create a Python script with the following content as `api_sample.py`:


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
```

#### 2. Generate TypeScript types

Run the following script:

```python3
import ts_type as ts
from api_sample import apis

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


ts.generator.add(Apis, 'apis', 'Apis')
ts.generator.generate('output_dir')
```

#### 3. Expected Output

Running the script will generate the following TypeScript definitions:
 
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

```typescript
type RequestForApiA = Extract<Apis, {url: "/api/a"}>['request'];
type ResponseForApiA = Extract<Apis, {url: "/api/a"}>['response'];
type RequestForApiB = Extract<Apis, {url: "/api/b"}>['request'];
type ResponseForApiB = Extract<Apis, {url: "/api/b"}>['response'];
```



### More Examples

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
    author: Author
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
    "author": example__Article__Author;
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
