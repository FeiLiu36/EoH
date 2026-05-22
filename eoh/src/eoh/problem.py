# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import ast
import sys
import types
import logging
import warnings
import itertools
from abc import ABC, abstractmethod

_module_counter = itertools.count()

logger = logging.getLogger('eoh')


def _extract_import_lines(template_program: str) -> str:
    """Return all top-level import statements from template_program as a single string."""
    tree = ast.parse(template_program)
    lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
    return "\n".join(lines)


def _detect_template_kind(template_program: str) -> str:
    """Return the structural kind of a template: 'class', 'multi_function', or 'function'."""
    tree = ast.parse(template_program)
    if any(isinstance(n, ast.ClassDef) for n in tree.body):
        return 'class'
    top_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    return 'multi_function' if len(top_funcs) > 1 else 'function'


def _get_entry_name(template_program: str) -> str:
    """Return the primary callable name to look up after exec().

    - Class template   → class name (first top-level ClassDef)
    - Multi-function   → last top-level function (typically the main entry point)
    - Single function  → function name
    """
    tree = ast.parse(template_program)
    top_classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    if top_classes:
        return top_classes[0]
    top_funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if top_funcs:
        return top_funcs[-1]
    raise ValueError("No function or class definition found in template_program.")


class BaseProblem(ABC):
    """Base class for all EoH problems.

    Evaluation settings (timeout, parallelism) are owned by the problem so
    each task controls its own computational budget.

    Example::

        class MyProblem(BaseProblem):
            template_program = '''
        def heuristic(x: np.ndarray) -> float:
            return float(x.mean())
        '''
            task_description = "Design a heuristic for ..."

            def evaluate_program(self, program_str, callable_func):
                return callable_func(my_input)

        task = MyProblem(timeout=30, n_processes=4)

    Attributes:
        template_program: Python source of the target function.
        task_description: One sentence describing the optimisation task.
        timeout:     Seconds allowed per evaluation (default 40).
        n_processes: Parallel evaluation workers (default 1, -1 = all CPUs).
    """

    template_program: str = ""
    task_description: str = ""

    def __init__(self, timeout: int = 40, n_processes: int = 1):
        import multiprocessing
        self.timeout = timeout
        self.n_processes = multiprocessing.cpu_count() if n_processes == -1 else n_processes

    def evaluate(self, code_string: str) -> float | None:
        """Called by the framework. Compiles code_string then delegates to evaluate_program."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import numpy as np
                module = types.ModuleType(f"heuristic_module_{next(_module_counter)}")
                import_prefix = _extract_import_lines(self.template_program)
                if import_prefix:
                    exec(import_prefix, module.__dict__)
                module.__dict__.setdefault('np', np)
                exec(code_string, module.__dict__)
                sys.modules[module.__name__] = module
                entry_name = _get_entry_name(self.template_program)
                callable_obj = getattr(module, entry_name, None)
                if callable_obj is None:
                    logger.debug("  [eval] '%s' not found in generated code", entry_name)
                    return None
                return self.evaluate_program(code_string, callable_obj)
        except Exception as e:
            logger.debug("  [eval] %s: %s", type(e).__name__, e, exc_info=True)
            return None

    @abstractmethod
    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        """Evaluate a generated algorithm.

        Args:
            program_str:   The full generated source code as a string.
            callable_func: The compiled target function, ready to call.

        Returns:
            A float fitness value (lower is better), or None if evaluation fails.
        """
        ...
