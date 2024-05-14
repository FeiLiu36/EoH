from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    """Configuration of a ProgramsDatabase.

    Attributes:
        functions_per_prompt: Number of previous programs to include in prompts.
        num_islands: Number of islands to maintain as a diversity mechanism.
        reset_period: How often (in seconds) the weakest islands should be reset.
        cluster_sampling_temperature_init: Initial temperature for softmax sampling
            of clusters within an island.
        cluster_sampling_temperature_period: Period of linear decay of the cluster
            sampling temperature.
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


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
    programs_database: ProgramsDatabaseConfig = dataclasses.field(default_factory=ProgramsDatabaseConfig)
    num_samplers: int = 4
    num_evaluators: int = 16
    samples_per_prompt: int = 4
