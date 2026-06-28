# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

from .utils.createFolders import create_folders
from .config import EoHConfig, LLMConfig
from .eoh.eoh import EOH


class EoH:
    """Entry point for the EoH framework.

    Usage::

        from eoh import EoH, LLMConfig, BaseProblem
        from my_problem import MyProblem

        llm  = LLMConfig(api_endpoint="...", api_key="...", model="...")
        task = MyProblem(timeout=40)

        eoh  = EoH(
            llm=llm, problem=task, pop_size=5, n_pop=20,
            num_samplers=16, num_evaluators=4,
        )
        eoh.run()

    Parallelism (async producer/consumer pipeline, decoupled from pop_size).
    These are direct EoH arguments — set them here, not on the problem or LLM:

        num_samplers    Number of concurrent LLM-generation threads (I/O bound).
                        The main throughput knob; independent of pop_size.
                        Default 1 (sequential). Pass -1 for all CPUs.
        num_evaluators  Number of concurrent evaluation subprocesses (CPU bound;
                        each eval is isolated in a hard-timeout subprocess).
                        Default 1. Pass -1 for all CPUs. Set
                        num_samplers >= num_evaluators to keep the eval pool busy.
        max_sample_nums Total evolution-sample budget (counts attempts, matching
                        the classic n_pop * pop_size generational run, which is
                        the default when unset).
        n_processes     Optional legacy shortcut: when given, sets both
                        num_samplers and num_evaluators (unless those are set
                        explicitly). Prefer the two explicit knobs above.

    Note: evolution is asynchronous steady-state — offspring register as soon as
    they finish rather than in synchronized generations, so runs are not
    bit-for-bit reproducible across different concurrency settings.
    """

    def __init__(
        self,
        llm: LLMConfig,
        problem,
        pop_size: int = 5,
        n_pop: int = 20,
        operators: list = None,
        operator_weights: list = None,
        n_parents: int = 2,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        max_sample_nums: int = None,
        n_processes: int = None,
        output_dir: str = "./",
        debug: bool = False,
        use_seed: bool = False,
        seed_path: str = "./seeds/seeds.json",
        use_continue: bool = False,
        continue_path: str = "./results/pops/population_generation_0.json",
        continue_id: int = 0,
    ):
        # num_samplers / num_evaluators are the direct concurrency controls for
        # EoH — they are NOT read from the problem or the LLM config. -1 → all
        # CPUs. n_processes is an optional legacy shortcut that sets both pools
        # at once when the explicit knobs are left at their defaults.
        import multiprocessing as _mp

        def _resolve(n):
            if n is None:
                return None
            return _mp.cpu_count() if n == -1 else max(1, int(n))

        n_processes = _resolve(n_processes)
        num_samplers = _resolve(num_samplers)
        num_evaluators = _resolve(num_evaluators)

        # Apply the legacy shortcut only where the explicit knob was left unset
        # (default 1). Explicit num_samplers / num_evaluators always win.
        if n_processes is not None:
            if num_samplers == 1:
                num_samplers = n_processes
            if num_evaluators == 1:
                num_evaluators = n_processes

        # Keep the problem's evaluation parallelism (used by the legacy seed
        # path) aligned with the evaluator pool size.
        problem.n_processes = num_evaluators

        config = EoHConfig(
            llm=llm,
            pop_size=pop_size,
            n_pop=n_pop,
            operators=operators if operators is not None else ['e1', 'e2', 'm1', 'm2'],
            operator_weights=operator_weights,
            n_parents=n_parents,
            num_samplers=num_samplers,
            num_evaluators=num_evaluators,
            max_sample_nums=max_sample_nums,
            output_dir=output_dir,
            debug=debug,
            use_seed=use_seed,
            seed_path=seed_path,
            use_continue=use_continue,
            continue_path=continue_path,
            continue_id=continue_id,
        )
        create_folders(config.output_dir)
        self._config = config
        self._problem = problem

    def run(self):
        EOH(self._config, self._problem).run()
