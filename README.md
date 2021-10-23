It generates typescript's type definitions from python's objects.


## Usage

1. add types to `TypeDefinitionGenerator`

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
```

2. call `TypeDefinitionGenerator.generate`

```py3
import ts_type as ts

if __name__ == '__main__':
    ts.generator.generate('output_dir')
```


## Examples

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

type example__Article__Type = ("news" | "essay");

type example__Article__Author = {
    "id": number;
    "name": string;
};

type example__Pagination<T> = {
    "items": T[];
    "next_cursor": (string | null);
};

type example__Response = {
    "status": number;
    "articles": example__Pagination<example__Article>;
};
```
