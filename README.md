It generates typescript's type definitions from python's typing


1. add type to generator

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

2. generate type definitions

```py3
import ts_type as ts

if __name__ == '__main__':
    ts.generator.generate('output_dir')
```
