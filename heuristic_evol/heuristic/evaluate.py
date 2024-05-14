from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from typing import Any

from .code import TextFunctionProgramConverter, Program
from .modify_code import ModifyCode


class Evaluator(ABC):
    def __init__(
            self,
            use_numba_accelerate: bool = False,
            use_protected_div: bool = False,
            protected_div_delta: float = 1e-5,
            random_seed: int | None = None,
            timeout_seconds: int | float = None,
            *,
            safe_evaluate: bool = True
    ):
        """Evaluator for executing generated code.
        Args:
            use_numba_accelerate: Wrapped the function with '@numba.jit(nopython=True)'.
            use_protected_div   : Modify 'a / b' => 'a / (b + delta)'.
            protected_div_delta : Delta value in protected div.
            random_seed         : If is not None, set random seed in the first line of the function body.
            timeout_seconds     : Terminate the evaluation after timeout seconds.
            safe_evaluate       : Evaluate in safe mode using a new process. If is set to False,
                the evaluation will not be terminated after timeout seconds. The user should consider how to
                terminate evaluating in time.

        -Assume that: use_numba_accelerate=True, self.use_protected_div=True, and self.random_seed=2024.
        -The original function:
        --------------------------------------------------------------------------------
        import numpy as np

        def f(a, b):
            a = np.random.random()
            return a / b
        --------------------------------------------------------------------------------
        -The modified function will be:
        --------------------------------------------------------------------------------
        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f():
            np.random.seed(2024)
            a = np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)
        --------------------------------------------------------------------------------
        As shown above, the 'import numba', 'numba.jit()' decorator, and '_protected_dev' will be added by this function.
        """
        self.use_numba_accelerate = use_numba_accelerate
        self.use_protected_div = use_protected_div
        self.protected_div_delta = protected_div_delta
        self.random_seed = random_seed
        self.timeout_seconds = timeout_seconds
        self.safe_evaluate = safe_evaluate

    @abstractmethod
    def evaluate_program(self, program_str: str, program_callable: callable) -> Any | None:
        r"""Evaluate a given function. You can use compiled function (function_callable),
        as well as the original function strings for evaluation.
        Args:
            program_str: The function in string. You can ignore this argument when implementation. (See below).
            program_callable: The callable python function.
        Return:
            Returns the fitness value.

        Assume that: self.use_numba_accelerate = True, self.use_protected_div = True,
        and self.random_seed = 2024, the argument 'function_str' will be something like below:
        --------------------------------------------------------------------------------
        import numpy as np
        import numba

        @numba.jit(nopython=True)
        def f(a, b):
            np.random.seed(2024)
            a = a + np.random.random()
            return _protected_div(a, b)

        def _protected_div(a, b, delta=1e-5):
            return a / (b + delta)
        --------------------------------------------------------------------------------
        As shown above, the 'import numba', 'numba.jit()' decorator,
        and '_protected_dev' will be added by this function.
        """
        raise NotImplementedError('Must provide a evaluator for a function.')


class SecureEvaluator:
    def __init__(self, evaluator: Evaluator, debug_mode=False):
        self._evaluator = evaluator
        self._debug_mode = debug_mode

    def _modify_program_code(self, program_str: str) -> str:
        function_name = TextFunctionProgramConverter.text_to_function(program_str).name
        if self._evaluator.use_numba_accelerate:
            program_str = ModifyCode.add_numba_decorator(
                program_str, function_name=function_name
            )
        if self._evaluator.use_protected_div:
            program_str = ModifyCode.replace_div_with_protected_div(
                program_str, self._evaluator.protected_div_delta, self._evaluator.use_numba_accelerate
            )
        if self._evaluator.random_seed is not None:
            program_str = ModifyCode.add_numpy_random_seed_to_func(
                program_str, function_name, self._evaluator.random_seed
            )
        return program_str

    def evaluate_program(self, program: str | Program):
        try:
            program_str = str(program)
            program_str = self._modify_program_code(program_str)
            if self._debug_mode:
                print(f'DEBUG: evaluated program:\n{program_str}\n')

            # safe evaluate
            if self._evaluator.safe_evaluate:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=self._evaluate_in_safe_process,
                    args=(program_str, result_queue)
                )
                process.start()
                # join to the current process after timeout seconds
                if self._evaluator.timeout_seconds is not None:
                    process.join(timeout=self._evaluator.timeout_seconds)
                else:
                    process.join()
                if process.is_alive():
                    if self._debug_mode:
                        print(f'DEBUG: the evaluation time exceeds {self._evaluator.timeout_seconds}s.')
                    # if the process is not finished in time
                    # we consider the program illegal
                    process.terminate()
                    process.join()
                    results = None
                else:
                    if not result_queue.empty():
                        results = result_queue.get_nowait()
                    else:
                        results = None
                return results
            else:
                return self._evaluate(program_str)
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None

    def _evaluate_in_safe_process(self, program_str: str, result_queue: multiprocessing.Queue):
        try:
            # get function name
            function_name = TextFunctionProgramConverter.text_to_function(program_str).name
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program_str, all_globals_namespace)
            # get the pointer of 'function_to_run'
            program_callable = all_globals_namespace[function_name]
            # get evaluate result
            res = self._evaluator.evaluate_program(program_str, program_callable)
            result_queue.put(res)
        except Exception as e:
            if self._debug_mode:
                print(e)
            result_queue.put(None)

    def _evaluate(self, program_str: str):
        try:
            # get function name
            function_name = TextFunctionProgramConverter.text_to_function(program_str).name
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program_str, all_globals_namespace)
            # get the pointer of 'function_to_run'
            program_callable = all_globals_namespace[function_name]
            # get evaluate result
            res = self._evaluator.evaluate_program(program_str, program_callable)
            return res
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None


class FakeEvaluatorForDebugging(Evaluator):
    def __init__(self):
        super().__init__(timeout_seconds=10)

    def evaluate_program(self, program_str: str, program_callable: callable) -> Any | None:
        return program_callable()
