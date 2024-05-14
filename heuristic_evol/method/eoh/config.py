from __future__ import annotations

import dataclasses
from typing import Type

from .sampler import EoHSampler
from ...heuristic import TensorboardProfiler, Evaluator

@dataclasses.dataclass(frozen=True)
class Config:
    pop_size: int = 50
    cx_rate: float = 0.7
    mut_rate: float = 0.2
    use_i1_operator: bool = True
    use_e1_operator: bool = True
    use_e2_operator: bool = True
    use_m1_operator: bool = True
    use_m2_operator: bool = True
    num_samplers: int = 4
    num_evaluators: int = 16
