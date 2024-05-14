from __future__ import annotations

import concurrent.futures
from threading import Thread

from . import programs_database
from .config import Config
from ...heuristic import *


class FunSearch:
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
        self._debug_mode = debug_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(template_program)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(template_program)

        # population, sampler, and evaluator
        self._database = programs_database.ProgramsDatabase(
            config.programs_database,
            self._template_program,
            self._function_to_evolve_name
        )
        self._sampler = sampler
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
        self._sampler_threads = [
            Thread(target=self._sample_evaluate_register) for _ in range(self._config.num_samplers)
        ]

    def _sample_evaluate_register(self):
        while self._max_sample_nums and self._tot_sample_nums < self._max_sample_nums:
            # get prompt
            prompt = self._database.get_prompt()
            prompt_contents = [prompt.code for _ in range(self._config.samples_per_prompt)]

            # do sample
            sampled_funcs = self._sampler.draw_samples(prompt_contents)

            # convert samples to program instances
            programs_to_be_eval = []
            for func in sampled_funcs:
                program = Sampler.sample_to_program(func, self._template_program)
                # if sample to program success
                if program is not None:
                    programs_to_be_eval.append(program)
                else:
                    # if convert failed, simply add a bad program
                    programs_to_be_eval.append(Program(preface=self._template_program.preface, functions=[]))

            # submit tasks to the thread pool and evaluate
            futures = []
            for program in programs_to_be_eval:
                future = self._evaluation_executor.submit(self._evaluator.evaluate_program, program)
                futures.append(future)
            scores = [f.result() for f in futures]

            # register to program database and profiler
            island_id = prompt.island_id
            for program, score in zip(programs_to_be_eval, scores):
                # convert to Function instance
                function = TextFunctionProgramConverter.program_to_function(program)
                # check if the function has converted to Function instance successfully
                if function is None:
                    continue
                # register to program database
                if score is not None:
                    self._database.register_function(
                        function=function,
                        island_id=island_id,
                        score=score
                    )
                # register to profiler
                if self._profiler is not None:
                    function.score = score
                    self._profiler.register_function(function)

            # update
            self._tot_sample_nums += self._config.samples_per_prompt

    def run(self):
        # evaluate the template program, make sure the score of which is not 'None'
        score = self._evaluator.evaluate_program(program=self._template_program)
        if score is None:
            raise RuntimeError('The score of the template function must not be "None".')

        # register the template program to the program database
        self._database.register_function(function=self._function_to_evolve, island_id=None, score=score)
        if self._profiler:
            self._function_to_evolve.score = score
            self._profiler.register_function(self._function_to_evolve)

        # start sampling using multiple threads
        for t in self._sampler_threads:
            t.start()

        # join all threads to the main thread
        for t in self._sampler_threads:
            t.join()

        if self._profiler is not None:
            self._profiler.finish()