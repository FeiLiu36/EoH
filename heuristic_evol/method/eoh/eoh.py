from __future__ import annotations

import concurrent.futures
from threading import Thread

from .config import Config
from .population import Population
from .prompt import EoHPrompt
from .sampler import EoHSampler

from ...heuristic import (
    Evaluator, ProfilerBase, InstructLLMSampler, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)


class EoH:
    def __init__(
            self,
            template_program: str,
            task_description: str,
            sampler: InstructLLMSampler,
            evaluator: Evaluator,
            profiler: ProfilerBase = None,
            config: Config = Config(),
            max_generations: int | None = 10,
            max_sample_nums: int | None = None,
            *,
            initial_sample_num: int | None = None,
            debug_mode: bool = False,
            multi_thread_or_process_eval: str = 'thread'
    ):
        # arguments and keywords
        self._template_program_str = template_program
        self._task_description_str = task_description
        self._config = config
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._debug_mode = debug_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(template_program)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(template_program)

        # population, sampler, and evaluator
        self._population = Population()
        self._sampler = EoHSampler(sampler, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluator, debug_mode=debug_mode)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0 if initial_sample_num is None else initial_sample_num

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._config.num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._config.num_evaluators)

        # threads for sampling
        self._sampler_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._config.num_samplers)

    def _init(self):
        pass
