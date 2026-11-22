# Vector add

## 范围

The three published L20D shapes use Model(alpha).forward(x, residual).
The 1024 element tile and two bfloat16 rounding boundaries are inherited
from the supplied proposal. Historical measurements do not establish new-run
correctness, performance or broader shape coverage.

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | One independent program per 1024 element tile; masked loads, two rounding boundaries, masked store | Sole design and shared Contract; no child designs or providers |
