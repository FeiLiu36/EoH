from __future__ import annotations

import concurrent.futures
import copy
from threading import Thread
from typing import Type

from .config import Config
from ...heuristic import *


class HillClimb:
    def __init__(
            self,
            template_program: str,
            sampler: Sampler,
            evaluator: Evaluator,
            profiler: ProfilerBase = None,
            config: Config = Config(),
            max_sample_nums: int | None = 20,
            *,
            initial_sample_num: int | None = None,
            debug_mode: bool = False,
            multi_thread_or_process_eval: str = 'thread'
    ):
        # arguments and keywords
        self._template_program_str = template_program
        self._config = config
        self._max_sample_nums = max_sample_nums
        self._debug_model = debug_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(template_program)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(template_program)

        # sampler and evaluator
        self._sampler = sampler
        self._evaluator = SecureEvaluator(evaluator, debug_mode=debug_mode)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0 if initial_sample_num is None else initial_sample_num
        self._best_function_found = self._function_to_evolve  # set to the template function at the beginning

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._config.num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._config.num_evaluators)

        # threads for sampling
        self._sampler_threads = [
            Thread(target=self._sample_evaluate_register) for _ in range(self._config.num_samplers)
        ]

    def _init(self):
        # evaluate the template program, make sure the score of which is not 'None'
        score = self._evaluator.evaluate_program(program=self._template_program)
        if score is None:
            raise RuntimeError('The score of the template function must not be "None".')
        self._best_function_found.score = score

        # register the template program to the program database
        if self._profiler:
            self._function_to_evolve.score = score
            self._profiler.register_function(self._function_to_evolve)

    def _get_prompt(self) -> str:
        template = TextFunctionProgramConverter.function_to_program(self._best_function_found, self._template_program)
        template.functions[0].name += '_v0'
        func_to_be_complete = copy.deepcopy(self._function_to_evolve)
        func_to_be_complete.name = self._function_to_evolve_name + '_v1'
        func_to_be_complete.docstring = f'  """Improved version of \'{self._function_to_evolve_name}_v0\'."""'
        func_to_be_complete.body = ''
        return '\n'.join([str(template), str(func_to_be_complete)])

    def _sample_evaluate_register(self):
        while self._max_sample_nums and self._tot_sample_nums < self._max_sample_nums:
            # do sample
            prompt_content = self._get_prompt()
            # print(prompt_content)
            sampled_funcs = self._sampler.draw_samples([prompt_content])

            # convert to program instance
            programs_to_be_eval = []
            for func in sampled_funcs:
                program = Sampler.sample_to_program(func, self._template_program)
                # if sample to program success
                if program is not None:
                    programs_to_be_eval.append(program)
                else:
                    # if convert failed, simply add a bad program
                    programs_to_be_eval.append(Program(preface=self._template_program.preface, functions=[]))

            # submit tasks to the thread pool
            futures = []
            for program in programs_to_be_eval:
                future = self._evaluation_executor.submit(self._evaluator.evaluate_program, program)
                futures.append(future)
            scores = [f.result() for f in futures]

            # update register to program database
            for program, score in zip(programs_to_be_eval, scores):
                # convert to Function instance
                function = TextFunctionProgramConverter.program_to_function(program)
                # check if the function has converted to Function instance successfully
                if function is None:
                    continue
                function.score = score
                # update best function found
                if score is not None and score > self._best_function_found.score:
                    self._best_function_found = function
                # register to profiler
                if self._profiler:
                    self._profiler.register_function(function)

            # update
            self._tot_sample_nums += 1

    def run(self):
        # do init
        self._init()

        # start sampling using multiple threads
        for t in self._sampler_threads:
            t.start()

        # join all threads to the main thread
        for t in self._sampler_threads:
            t.join()

        if self._profiler is not None:
            self._profiler.finish()