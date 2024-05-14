from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration of a FunSearch experiment.

    Attributes:
        programs_database: Configuration of the evolutionary algorithm.
        num_samplers: Number of independent Samplers in the experiment. A value
            larger than 1 only has an effect when the samplers are able to execute
            in parallel, e.g. on different machines of a distributed system.
        num_evaluators: Number of independent program Evaluators in the experiment.
            A value larger than 1 is only expected to be useful when the Evaluators
            can execute in parallel as part of a distributed system.
        samples_per_prompt: How many independently sampled program continuations to
            obtain for each prompt.
    """
    num_samplers: int = 4
    num_evaluators: int = 16
